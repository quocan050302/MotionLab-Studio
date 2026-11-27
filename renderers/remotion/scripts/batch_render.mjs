/**
 * Phase R1-A7 (ML-01.R5.R1): Offline Single-Bundle, Persistent-Browser Remotion Batch Renderer.
 *
 * Uses ALREADY-INSTALLED local `@remotion/bundler` and `@remotion/renderer` (4.0.547):
 *   1. Content-addressed Webpack bundle cache keyed by TSX/dependency code hash (0 re-bundles per scene).
 *   2. Single `openBrowser()` instance reused across all scene MP4s and active-window ablation stills.
 *   3. Hardlinks per-scene media under `<serveDir>/public/<sceneId>/` so zero media duplication or state leaks occur.
 *   4. Always closes the owned Chromium instance cleanly in `finally` (never kills external Chrome processes).
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { bundle } = require("@remotion/bundler");
const { openBrowser, renderMedia, renderStill, selectComposition } = require("@remotion/renderer");

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--manifest=")) {
      out.manifest = a.slice("--manifest=".length);
    } else if (a === "--manifest" && i + 1 < argv.length) {
      out.manifest = argv[++i];
    }
  }
  return out;
}

function linkOrCopy(src, dst) {
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  if (fs.existsSync(dst)) {
    fs.unlinkSync(dst);
  }
  try {
    fs.linkSync(src, dst);
  } catch {
    fs.copyFileSync(src, dst);
  }
}

function copyDirShallowRecursive(srcDir, dstDir) {
  fs.mkdirSync(dstDir, { recursive: true });
  for (const ent of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const s = path.join(srcDir, ent.name);
    const d = path.join(dstDir, ent.name);
    if (ent.isDirectory()) {
      copyDirShallowRecursive(s, d);
    } else if (ent.isFile()) {
      linkOrCopy(s, d);
    }
  }
}

async function main() {
  const tStart = performance.now();
  const args = parseArgs(process.argv);
  if (!args.manifest || !fs.existsSync(args.manifest)) {
    throw new Error(`Missing --manifest path: ${args.manifest}`);
  }
  const spec = JSON.parse(fs.readFileSync(args.manifest, "utf-8"));
  const {
    entryPoint,
    bundleCacheDir,
    codeHash,
    serveDir,
    browserExecutable,
    publicMounts = [],
    jobs = [],
    resultPath,
  } = spec;

  // 1. Reuse content-addressed code bundle if valid; otherwise bundle ONCE
  const tBundle0 = performance.now();
  const metaPath = path.join(bundleCacheDir, "bundle_meta.json");
  let bundleReused = false;
  if (
    fs.existsSync(path.join(bundleCacheDir, "index.html")) &&
    fs.existsSync(metaPath)
  ) {
    try {
      const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
      if (meta.codeHash === codeHash) {
        bundleReused = true;
      }
    } catch {
      bundleReused = false;
    }
  }

  if (!bundleReused) {
    fs.rmSync(bundleCacheDir, { recursive: true, force: true });
    fs.mkdirSync(bundleCacheDir, { recursive: true });
    const emptyPub = path.join(bundleCacheDir, "_empty_pub");
    fs.mkdirSync(emptyPub, { recursive: true });
    await bundle({
      entryPoint: path.resolve(entryPoint),
      outDir: path.resolve(bundleCacheDir),
      publicDir: path.resolve(emptyPub),
      enableCaching: false,
    });
    fs.rmSync(emptyPub, { recursive: true, force: true });
    fs.writeFileSync(
      metaPath,
      JSON.stringify({ codeHash, createdAt: new Date().toISOString() }, null, 2),
      "utf-8"
    );
  }
  const bundleS = Number(((performance.now() - tBundle0) / 1000).toFixed(3));

  // 2. Materialize lightweight isolated serveDir via hardlinks + per-scene public mounts
  fs.rmSync(serveDir, { recursive: true, force: true });
  copyDirShallowRecursive(bundleCacheDir, serveDir);
  const pubRoot = path.join(serveDir, "public");
  fs.mkdirSync(pubRoot, { recursive: true });
  for (const m of publicMounts) {
    const src = path.resolve(m.src);
    const rel = String(m.rel).replace(/^[/\\]+/, "");
    const dst = path.join(pubRoot, rel);
    if (fs.existsSync(src)) {
      linkOrCopy(src, dst);
    }
  }

  // 3. Launch ONE persistent local Chromium browser for all scenes & ablation stills
  const tBrowser0 = performance.now();
  const browser = await openBrowser("chrome", {
    browserExecutable: browserExecutable || undefined,
    logLevel: "error",
  });
  const browserOpenS = Number(((performance.now() - tBrowser0) / 1000).toFixed(3));

  const jobResults = [];
  try {
    for (const job of jobs) {
      const tj0 = performance.now();
      const inputProps = job.props || {};
      const composition = await selectComposition({
        serveUrl: path.resolve(serveDir),
        id: job.compositionId,
        inputProps,
        puppeteerInstance: browser,
        logLevel: "error",
      });

      fs.mkdirSync(path.dirname(path.resolve(job.outputLocation)), { recursive: true });

      if (job.kind === "still") {
        await renderStill({
          composition,
          serveUrl: path.resolve(serveDir),
          output: path.resolve(job.outputLocation),
          inputProps,
          frame: Number(job.frame || 0),
          imageFormat: "png",
          puppeteerInstance: browser,
          logLevel: "error",
        });
      } else {
        const codec = job.codec || "h264";
        await renderMedia({
          composition,
          serveUrl: path.resolve(serveDir),
          codec,
          pixelFormat: job.pixelFormat || "yuv420p",
          crf: codec === "h264" ? Number(job.crf ?? 16) : undefined,
          imageFormat: codec === "vp9" ? "png" : "jpeg",
          muted: true,
          concurrency: Number(job.concurrency ?? 1),
          outputLocation: path.resolve(job.outputLocation),
          inputProps,
          puppeteerInstance: browser,
          logLevel: "error",
        });
      }

      const elapsedS = Number(((performance.now() - tj0) / 1000).toFixed(3));
      jobResults.push({
        id: job.id,
        kind: job.kind || "media",
        outputLocation: path.resolve(job.outputLocation),
        elapsed_s: elapsedS,
        status: "OK",
      });
    }
  } finally {
    try {
      await browser.close({ silent: true });
    } catch {
      // Ignore cleanup race if browser already exited
    }
    // Remove transient serveDir (hardlinks only; never touches bundleCacheDir or original media)
    try {
      fs.rmSync(serveDir, { recursive: true, force: true });
    } catch {
      // Best-effort cleanup
    }
  }

  const summary = {
    status: "OK",
    code_hash: codeHash,
    bundle_reused: bundleReused,
    bundle_s: bundleS,
    browser_open_s: browserOpenS,
    browser_launches: 1,
    total_s: Number(((performance.now() - tStart) / 1000).toFixed(3)),
    jobs: jobResults,
  };
  if (resultPath) {
    fs.writeFileSync(path.resolve(resultPath), JSON.stringify(summary, null, 2), "utf-8");
  }
  process.stdout.write(JSON.stringify(summary) + "\n");
}

main().catch((err) => {
  process.stderr.write(String(err?.stack || err) + "\n");
  process.exit(1);
});

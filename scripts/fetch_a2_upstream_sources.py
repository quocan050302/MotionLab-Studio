"""Phase A2 Bootstrap: Fetch real verified upstream HyperFrames blocks (via local pinned CLI)
and exact pinned-SHA MIT Remotion component source files into staging for inspection & import.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".cache" / "repo_staging"
HF_STAGE = STAGE / "hf_blocks_stage"
REM_STAGE = STAGE / "remotion_sources_stage"
HF_BIN = ROOT / "renderers" / "hyperframes" / "node_modules" / ".bin" / ("hyperframes.cmd" if os.name == "nt" else "hyperframes")

HF_BLOCKS = [
    "apple-money-count",
    "count-up",
    "data-chart",
    "mk-progress-stat",
    "strikethrough-replace",
    "kinetic-center-build",
    "mk-callout-highlight",
    "mk-specs-list",
    "exploded-stack",
    "organic-light-leak-overlay",
    "lt-mask-reveal",
    "flowchart-vertical",
]

REM_REPOS = [
    ("R02", "riaz37/remotion-ui", "27e51ae809"),
    ("R04", "lifeprompt-team/remotion-scenes", "02c7a84241"),
    ("R06", "vivmagarwal/remotion-effects-library", "f2781731e1"),
    ("R08", "RenderComp/free-remotion-templates", "f648980e52"),
    ("R10", "GTKottman/remotion-transitions", "58df899996"),
    ("R01", "Remocn/remocn", "52ec1ffe46"),
]


def gh_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MotionLab-R5-A2/1.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_hf_blocks() -> dict:
    HF_STAGE.mkdir(parents=True, exist_ok=True)
    # Initialize minimal project marker so `hyperframes add` installs cleanly inside HF_STAGE
    if not (HF_STAGE / "index.html").is_file():
        (HF_STAGE / "index.html").write_text(
            '<!doctype html><html><body><div id="root" data-composition-id="stage" data-width="720" data-height="1280" data-fps="24" data-duration="4"></div></body></html>\n',
            encoding="utf-8",
        )
        (HF_STAGE / "hyperframes.json").write_text('{"version": 1, "entry": "index.html"}\n', encoding="utf-8")

    results = {}
    for blk in HF_BLOCKS:
        cmd = [str(HF_BIN), "add", blk, "--yes"]
        proc = subprocess.run(cmd, cwd=str(HF_STAGE), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if proc.returncode != 0:
            # Retry without --yes in case CLI flag differs
            proc = subprocess.run([str(HF_BIN), "add", blk], cwd=str(HF_STAGE), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        results[blk] = {
            "rc": proc.returncode,
            "stdout": (proc.stdout or "").strip()[-400:],
            "stderr": (proc.stderr or "").strip()[-240:],
        }
        print(f"HF ADD {blk:28s} rc={proc.returncode} | {(proc.stdout or proc.stderr).strip().splitlines()[-1:] if (proc.stdout or proc.stderr).strip() else ''}", flush=True)

    # List installed files in HF_STAGE
    files = sorted(str(p.relative_to(HF_STAGE)).replace("\\", "/") for p in HF_STAGE.rglob("*") if p.is_file())
    out = {"results": results, "installed_files": files}
    (STAGE / "hf_add_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"HF_STAGE installed files count: {len(files)}")
    return out


def inspect_remotion_trees() -> dict:
    REM_STAGE.mkdir(parents=True, exist_ok=True)
    survey = {r["code"]: r for r in json.loads((STAGE / "github_survey_verified.json").read_text(encoding="utf-8"))}
    trees_summary = {}
    for code, slug, _short in REM_REPOS:
        full_sha = survey[code]["head_sha"]
        print(f"\n=== TREE {code} {slug} @ {full_sha[:10]} ===", flush=True)
        try:
            tree = gh_json(f"https://api.github.com/repos/{slug}/git/trees/{full_sha}?recursive=1")
            paths = [it["path"] for it in tree.get("tree", []) if it.get("type") == "blob"]
            tsx_paths = [p for p in paths if p.endswith((".tsx", ".ts")) and "node_modules" not in p]
            trees_summary[code] = {
                "repo": slug,
                "sha": full_sha,
                "total_files": len(paths),
                "tsx_count": len(tsx_paths),
                "sample_paths": tsx_paths[:120],
                "all_tsx_paths": tsx_paths,
            }
            # Print relevant matches
            kws = ("waterfall", "bar-chart", "barchart", "counter", "slot", "strike", "matte", "parallax", "card-stack", "cardstack", "callout", "spotlight", "masked", "kinetic", "flip", "page", "particle", "morph", "liquid", "line-led", "lineled", "paper", "device", "product", "kpi", "textmask", "text-mask")
            hits = [p for p in tsx_paths if any(k in p.lower() for k in kws)]
            print(f"  total_tsx={len(tsx_paths)}, relevant_hits={len(hits)}")
            for h in hits[:25]:
                print(f"    {h}")
        except Exception as exc:
            trees_summary[code] = {"repo": slug, "sha": full_sha, "error": str(exc)}
            print(f"  ERROR: {exc}")

    (STAGE / "remotion_trees.json").write_text(json.dumps(trees_summary, indent=2), encoding="utf-8")
    return trees_summary


if __name__ == "__main__":
    fetch_hf_blocks()
    inspect_remotion_trees()


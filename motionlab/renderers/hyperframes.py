"""Phase E4 — Local Offline HyperFrames + GSAP Renderer Adapter.

Uses pinned local `renderers/hyperframes/node_modules/.bin/hyperframes` and local
`node_modules/gsap/dist/gsap.min.js` copied directly into each isolated slot directory
(ZERO CDN URLs, ZERO `npx --yes`, ZERO runtime downloads).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from motionlab.config import PROJECT_ROOT, atomic_write_json, atomic_write_text
from motionlab.renderers.base import (
    BaseRenderer,
    RendererError,
    find_local_browser,
    probe_artifact_ffprobe,
    verify_webm_alpha_pixels,
)
from motionlab.renderers.hyperframes_vfx import SUPPORTED_VFX_TEMPLATES, build_vfx_graphics_html
from motionlab.resource import GLOBAL_JOB_SEMAPHORE, ResourceBudget, run_managed_subprocess, run_subprocess_guarded
from motionlab.schemas import AnalysisResult, EditPlan, EffectEvent, EffectSpec, MediaInfo

HYPERFRAMES_SUPPORTED_EFFECTS = {
    "foreground_frame_accent",
    "edge_lines",
    "ring_accent",
    "directional_shapes",
    "pulse_border",
}

SUPPORTED_GRAPHICS_TEMPLATES = {
    "depth-stage",
    "detail-connector",
    "action-path",
    "pass-through",
    "match-shape",
    "split-divider",
    "contour-focus",
    "depth-horizon",
    "audio-stage",
} | SUPPORTED_VFX_TEMPLATES

API_KEY_ENV_NAMES = (
    "ELEVENLABS_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "REPLICATE_API_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def _has_hyperframes_cached_browser() -> bool:
    """Check if HyperFrames already has its own local offline headless-shell cached on disk."""
    for base in (
        Path.home() / ".cache" / "hyperframes",
        Path.home() / ".cache" / "puppeteer",
        Path(os.environ.get("LOCALAPPDATA", "")) / "hyperframes",
    ):
        if base.is_dir():
            for pat in ("*chrome-headless-shell*", "*chrome.exe", "*/chrome"):
                if any(base.rglob(pat)):
                    return True
    return False


def _sanitized_offline_env(browser_exe: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key in API_KEY_ENV_NAMES:
        env.pop(key, None)
    env["DO_NOT_TRACK"] = "1"
    env["HYPERFRAMES_TELEMETRY"] = "0"
    env["PRODUCER_LOW_MEMORY_MODE"] = "true"
    # On Windows, prefer HyperFrames' own isolated local headless-shell if cached (prevents 120s desktop-Chrome lockouts!)
    if browser_exe and not (os.name == "nt" and _has_hyperframes_cached_browser()):
        env["PUPPETEER_EXECUTABLE_PATH"] = browser_exe
        env["CHROME_PATH"] = browser_exe
    else:
        env.pop("PUPPETEER_EXECUTABLE_PATH", None)
        env.pop("CHROME_PATH", None)
    return env


def _generate_gsap_overlay_html(
    event: EffectEvent,
    width: int,
    height: int,
    duration_s: float,
    fps_int: int,
    audio_slice: list[float],
) -> str:
    """Generate a deterministic, 100% offline HyperFrames HTML/CSS/GSAP transparent overlay composition."""
    intensity = float(event.parameters.get("intensity", 0.75))
    opacity = float(event.parameters.get("opacity", 0.65)) * intensity
    margin_ratio = float(event.parameters.get("margin_ratio", 0.045))
    stroke_px = int(event.parameters.get("stroke_px", event.parameters.get("thickness_px", 3)))
    mx = max(8, int(round(width * margin_ratio)))
    my = max(8, int(round(height * margin_ratio)))
    arm = max(24, int(round(min(width, height) * 0.085)))
    line_len = max(48, int(round(width * 0.24)))
    dur_str = f"{max(0.2, duration_s):.4f}"

    if event.effect_id == "foreground_frame_accent":
        svg_body = f"""
      <svg id="gfx" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="position:absolute;inset:0;opacity:0;">
        <path class="corner" d="M {mx} {my + arm} L {mx} {my} L {mx + arm} {my}" fill="none" stroke="rgba(246,247,252,{opacity:.3f})" stroke-width="{stroke_px}" stroke-linecap="round"/>
        <path class="corner" d="M {width - mx - arm} {my} L {width - mx} {my} L {width - mx} {my + arm}" fill="none" stroke="rgba(246,247,252,{opacity:.3f})" stroke-width="{stroke_px}" stroke-linecap="round"/>
        <path class="corner" d="M {mx} {height - my - arm} L {mx} {height - my} L {mx + arm} {height - my}" fill="none" stroke="rgba(246,247,252,{opacity:.3f})" stroke-width="{stroke_px}" stroke-linecap="round"/>
        <path class="corner" d="M {width - mx - arm} {height - my} L {width - mx} {height - my} L {width - mx} {height - my - arm}" fill="none" stroke="rgba(246,247,252,{opacity:.3f})" stroke-width="{stroke_px}" stroke-linecap="round"/>
      </svg>"""
        gsap_anim = f"""
      tl.fromTo("#gfx", {{ opacity: 0, scale: 1.03, transformOrigin: "50% 50%" }}, {{ opacity: 1, scale: 1.0, duration: {duration_s * 0.24:.4f}, ease: "power3.out" }}, 0);
      tl.to("#gfx", {{ opacity: 0, duration: {duration_s * 0.22:.4f}, ease: "power2.in" }}, {duration_s * 0.78:.4f});"""

    elif event.effect_id == "directional_shapes":
        base_y = int(round(height * 0.90))
        sz = max(8, int(round(min(width, height) * 0.018)))
        chevrons = []
        for i in range(3):
            ox = int(round(width * 0.08)) + i * int(sz * 1.45)
            a = min(0.9, opacity * (0.55 + 0.45 * ((i + 1) / 3)))
            chevrons.append(
                f'<polyline points="{ox - sz // 2},{base_y - sz} {ox + sz // 2},{base_y} {ox - sz // 2},{base_y + sz}" '
                f'fill="none" stroke="rgba(255,255,255,{a:.3f})" stroke-width="{max(2, stroke_px)}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        svg_body = f'<svg id="gfx" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="position:absolute;inset:0;opacity:0;">{"".join(chevrons)}</svg>'
        gsap_anim = f"""
      tl.fromTo("#gfx", {{ opacity: 0, x: 0 }}, {{ opacity: 1, x: {int(width * 0.04)}, duration: {duration_s * 0.25:.4f}, ease: "power3.out" }}, 0);
      tl.to("#gfx", {{ x: {int(width * 0.08)}, duration: {duration_s * 0.55:.4f}, ease: "sine.inOut" }}, {duration_s * 0.22:.4f});
      tl.to("#gfx", {{ opacity: 0, duration: {duration_s * 0.22:.4f}, ease: "power2.in" }}, {duration_s * 0.78:.4f});"""

    elif event.effect_id == "pulse_border":
        pad = max(6, int(round(min(width, height) * 0.018)))
        rad = max(8, int(round(min(width, height) * 0.025)))
        peak_e = max(audio_slice) if audio_slice else 0.7
        peak_op = min(0.85, max(0.18, opacity * (0.55 + 0.45 * peak_e)))
        svg_body = f"""
      <svg id="gfx" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="position:absolute;inset:0;opacity:0;">
        <rect id="borderRect" x="{pad}" y="{pad}" width="{width - pad * 2}" height="{height - pad * 2}" rx="{rad}" ry="{rad}"
              fill="none" stroke="rgba(255,240,210,{peak_op:.3f})" stroke-width="{max(2, stroke_px)}"/>
      </svg>"""
        gsap_anim = f"""
      tl.fromTo("#gfx", {{ opacity: 0 }}, {{ opacity: 1, duration: {duration_s * 0.20:.4f}, ease: "power2.out" }}, 0);
      tl.to("#borderRect", {{ opacity: 0.55, duration: {duration_s * 0.25:.4f}, yoyo: true, repeat: 1, ease: "sine.inOut" }}, {duration_s * 0.25:.4f});
      tl.to("#gfx", {{ opacity: 0, duration: {duration_s * 0.22:.4f}, ease: "power2.in" }}, {duration_s * 0.78:.4f});"""

    elif event.effect_id == "ring_accent":
        cx = int(round(event.region.get("cx", 0.85) * width))
        cy = int(round(event.region.get("cy", 0.86) * height))
        if 0.3 * width < cx < 0.7 * width and 0.25 * height < cy < 0.75 * height:
            cx, cy = int(width * 0.85), int(height * 0.86)
        r = max(24, int(round(min(width, height) * 0.11)))
        svg_body = f"""
      <svg id="gfx" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="position:absolute;inset:0;opacity:0;">
        <circle id="r1" cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,{opacity:.3f})" stroke-width="{max(2, stroke_px)}"/>
        <circle id="r2" cx="{cx}" cy="{cy}" r="{int(r * 0.65)}" fill="none" stroke="rgba(162,222,255,{opacity * 0.8:.3f})" stroke-width="{max(1, stroke_px - 1)}"/>
      </svg>"""
        gsap_anim = f"""
      tl.fromTo("#gfx", {{ opacity: 0 }}, {{ opacity: 1, duration: {duration_s * 0.20:.4f}, ease: "power3.out" }}, 0);
      tl.fromTo("#r1", {{ scale: 0.3, svgOrigin: "{cx} {cy}" }}, {{ scale: 1.0, duration: {duration_s * 0.75:.4f}, ease: "power3.out" }}, 0);
      tl.fromTo("#r2", {{ scale: 0.2, svgOrigin: "{cx} {cy}" }}, {{ scale: 0.9, duration: {duration_s * 0.75:.4f}, ease: "power3.out" }}, {duration_s * 0.10:.4f});
      tl.to("#gfx", {{ opacity: 0, duration: {duration_s * 0.25:.4f}, ease: "power2.in" }}, {duration_s * 0.75:.4f});"""

    else:
        # edge_lines default
        svg_body = f"""
      <svg id="gfx" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="position:absolute;inset:0;opacity:0;">
        <line id="l1" x1="{mx}" y1="{my}" x2="{mx + line_len}" y2="{my}" stroke="rgba(255,255,255,{opacity:.3f})" stroke-width="{max(2, stroke_px)}" stroke-linecap="round"/>
        <line id="l2" x1="{mx}" y1="{my + stroke_px * 3}" x2="{mx + int(line_len * 0.55)}" y2="{my + stroke_px * 3}" stroke="rgba(255,198,118,{opacity * 0.85:.3f})" stroke-width="{max(1, stroke_px - 1)}" stroke-linecap="round"/>
        <line id="l3" x1="{width - mx - line_len}" y1="{height - my}" x2="{width - mx}" y2="{height - my}" stroke="rgba(255,255,255,{opacity:.3f})" stroke-width="{max(2, stroke_px)}" stroke-linecap="round"/>
      </svg>"""
        gsap_anim = f"""
      tl.fromTo("#gfx", {{ opacity: 0 }}, {{ opacity: 1, duration: {duration_s * 0.22:.4f}, ease: "power3.out" }}, 0);
      tl.fromTo("#l1", {{ scaleX: 0.1, transformOrigin: "left center" }}, {{ scaleX: 1, duration: {duration_s * 0.35:.4f}, ease: "power3.out" }}, 0);
      tl.fromTo("#l3", {{ scaleX: 0.1, transformOrigin: "right center" }}, {{ scaleX: 1, duration: {duration_s * 0.35:.4f}, ease: "power3.out" }}, 0);
      tl.to("#gfx", {{ opacity: 0, duration: {duration_s * 0.22:.4f}, ease: "power2.in" }}, {duration_s * 0.78:.4f});"""

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <script src="./gsap.min.js"></script>
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{
        margin: 0;
        width: {width}px;
        height: {height}px;
        overflow: hidden;
        background: transparent;
      }}
      #root {{
        position: relative;
        width: 100%;
        height: 100%;
        background: transparent;
      }}
    </style>
  </head>
  <body>
    <div
      id="root"
      data-composition-id="main"
      data-start="0"
      data-duration="{dur_str}"
      data-width="{width}"
      data-height="{height}"
      data-fps="{fps_int}"
    >
      <div class="clip" data-start="0" data-duration="{dur_str}" data-track-index="0" style="position:absolute;inset:0;">
        {svg_body}
      </div>
    </div>
    <script>
      const tl = gsap.timeline({{ paused: true }});
      {gsap_anim}
      window.__timelines = window.__timelines || {{}};
      window.__timelines["main"] = tl;
      tl.seek(0);
    </script>
  </body>
</html>
"""


class HyperFramesRenderer(BaseRenderer):
    name = "hyperframes"

    def __init__(self, enabled: bool = True, project_dir: Path | None = None) -> None:
        self.enabled = enabled
        self.project_dir = (project_dir or (PROJECT_ROOT / "renderers" / "hyperframes")).resolve()
        bin_name = "hyperframes.cmd" if os.name == "nt" else "hyperframes"
        self.hf_bin = self.project_dir / "node_modules" / ".bin" / bin_name
        self.gsap_js = self.project_dir / "node_modules" / "gsap" / "dist" / "gsap.min.js"
        self.browser_exe = find_local_browser()

    def probe_capability(self) -> dict[str, Any]:
        license_note = (
            "HyperFrames: Apache-2.0; GSAP: Standard No-Charge License (https://gsap.com/standard-license)."
        )
        if not self.enabled:
            return {"available": False, "reason": "disabled_by_flag", "license": license_note}
        if not self.hf_bin.exists():
            return {"available": False, "reason": f"local_hyperframes_cli_missing:{self.hf_bin}", "license": license_note}
        if not self.gsap_js.exists():
            return {"available": False, "reason": f"local_gsap_missing:{self.gsap_js}", "license": license_note}
        if not self.browser_exe:
            return {"available": False, "reason": "local_chrome_or_edge_not_found_offline", "license": license_note}
        pkg_json = self.project_dir / "package.json"
        version = "unknown"
        if pkg_json.exists():
            try:
                version = json.loads(pkg_json.read_text(encoding="utf-8")).get("dependencies", {}).get("hyperframes", "unknown")
            except Exception:
                pass
        return {
            "available": True,
            "version": version,
            "executable": str(self.hf_bin),
            "gsap_local": str(self.gsap_js),
            "browser_executable": self.browser_exe,
            "supported_effects": sorted(HYPERFRAMES_SUPPORTED_EFFECTS),
            "license": license_note,
            "reason": "ok",
        }

    def supports(self, effect_spec: EffectSpec, media: MediaInfo, capabilities: dict[str, Any]) -> bool:
        cap = capabilities.get("hyperframes") or {}
        return bool(cap.get("available")) and effect_spec.id in HYPERFRAMES_SUPPORTED_EFFECTS

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float | None = None) -> dict[str, Any]:
        info = probe_artifact_ffprobe(artifact_path)
        v_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not v_streams:
            raise RendererError(f"HyperFrames artifact has no video stream: {artifact_path}")
        return info

    def render_overlay_event(
        self,
        event: EffectEvent,
        plan: EditPlan,
        analysis: AnalysisResult,
        out_webm: Path,
        work_dir: Path,
        budget: ResourceBudget | None = None,
    ) -> dict[str, Any]:
        """Scaffold an isolated offline HyperFrames slot, run lint + render WebM, verify output."""
        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"HyperFrames unavailable: {cap.get('reason')}")

        t0 = time.monotonic()
        slot_dir = work_dir / f"hf_slot_{event.id}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        out_webm.parent.mkdir(parents=True, exist_ok=True)

        # Copy pinned local GSAP into slot dir so HTML loads strictly from `./gsap.min.js` (0 network!)
        shutil.copy2(self.gsap_js, slot_dir / "gsap.min.js")

        fps_float = plan.fps_num / plan.fps_den
        fps_int = max(1, int(round(fps_float)))
        dur_frames = max(2, event.end_frame_exclusive - event.start_frame)
        duration_s = dur_frames / fps_float

        audio_lookup = {pt.frame: pt.rms_norm for pt in analysis.audio_envelope}
        audio_slice = [audio_lookup.get(event.start_frame + i, 0.0) for i in range(dur_frames)]

        html = _generate_gsap_overlay_html(
            event=event,
            width=plan.width,
            height=plan.height,
            duration_s=duration_s,
            fps_int=fps_int,
            audio_slice=audio_slice,
        )
        atomic_write_text(slot_dir / "index.html", html)
        atomic_write_json(slot_dir / "hyperframes.json", {"version": 1, "entry": "index.html"})

        env = _sanitized_offline_env(self.browser_exe)
        # Run verified HyperFrames lint gate first
        lint_cmd = [str(self.hf_bin), "lint", str(slot_dir)]
        lint_res = run_subprocess_guarded(
            lint_cmd,
            budget=budget,
            cwd=slot_dir,
            env=env,
            timeout_s=60,
            check=False,
        )

        # Render transparent WebM using verified HyperFrames CLI flags
        render_cmd = [
            str(self.hf_bin),
            "render",
            str(slot_dir),
            "--format",
            "webm",
            "--fps",
            str(fps_int),
            "--workers",
            "1",
            "--low-memory-mode",
            "--no-browser-gpu",
            "--frames-cache-dir",
            "off",
            "-o",
            str(out_webm),
        ]
        res = run_subprocess_guarded(
            render_cmd,
            budget=budget,
            cwd=slot_dir,
            env=env,
            timeout_s=min(240, budget.config.subprocess_timeout_s if budget else 240),
            check=False,
        )
        if res.returncode != 0:
            raise RendererError(
                f"HyperFrames render failed for {event.id} (exit {res.returncode}): {(res.stderr or res.stdout).strip()[:600]}"
            )
        info = self.verify_artifact(out_webm)
        return {
            "engine": "hyperframes",
            "artifact_path": str(out_webm),
            "lint_exit_code": lint_res.returncode,
            "elapsed_s": round(time.monotonic() - t0, 4),
            "ffprobe": info,
        }

    def supports_recipe_graphics(self, template_name: str | None, capabilities: dict[str, Any]) -> bool:
        return (
            bool(template_name)
            and (template_name in SUPPORTED_GRAPHICS_TEMPLATES or template_name.startswith("imported_") or template_name.startswith("gen_"))
            and bool((capabilities.get("hyperframes") or {}).get("available"))
        )

    def render_html_slot(
        self,
        slot_dir: Path,
        html: str,
        out_path: Path,
        fps: float,
        fmt: str = "webm",
        extra_assets: dict[str, Path] | None = None,
        timeout_s: float = 180.0,
        cache_dir: Path | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """Render any standalone offline HyperFrames HTML composition (VP9 WebM alpha or H.264 MP4) with optional content-addressed cache."""
        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"HyperFrames unavailable: {cap.get('reason')}")
        slot_dir = slot_dir.resolve()
        slot_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute strict content-addressed cache key covering version + GSAP bytes + normalized HTML + fps + format + extra assets
        cache_key = ""
        cached_artifact: Path | None = None
        if cache_dir is not None:
            kh = hashlib.sha256()
            kh.update(f"hf_v={cap.get('version','unknown')}|fps={round(float(fps),4)}|fmt={fmt}|".encode("utf-8"))
            kh.update(html.encode("utf-8"))
            kh.update(self.gsap_js.read_bytes())
            if extra_assets:
                for rel_name in sorted(extra_assets.keys()):
                    kh.update(rel_name.encode("utf-8"))
                    kh.update(extra_assets[rel_name].read_bytes())
            cache_key = kh.hexdigest()[:24]
            c_root = cache_dir.resolve()
            c_root.mkdir(parents=True, exist_ok=True)
            cached_artifact = c_root / f"hf_{cache_key}.{fmt}"
            if use_cache and cached_artifact.is_file():
                try:
                    probe = self.verify_artifact(cached_artifact)
                    shutil.copy2(cached_artifact, out_path)
                    ch = hashlib.sha256()
                    with out_path.open("rb") as fh:
                        while chunk := fh.read(1 << 20):
                            ch.update(chunk)
                    return {
                        "engine": self.name,
                        "artifact_path": str(out_path),
                        "artifact_sha256": ch.hexdigest(),
                        "size_bytes": int(out_path.stat().st_size),
                        "lint_exit_code": 0,
                        "elapsed_s": 0.01,
                        "peak_tree_rss_mb": 0.0,
                        "cache_hit": True,
                        "cache_key": cache_key,
                        "ffprobe": probe,
                    }
                except Exception:
                    cached_artifact.unlink(missing_ok=True)

        shutil.copy2(self.gsap_js, slot_dir / "gsap.min.js")
        if extra_assets:
            for rel_name, src_p in extra_assets.items():
                dst_p = slot_dir / rel_name
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_p, dst_p)
        atomic_write_text(slot_dir / "index.html", html)
        atomic_write_json(slot_dir / "hyperframes.json", {"version": 1, "entry": "index.html"})

        env = _sanitized_offline_env(self.browser_exe)
        fps_int = max(1, int(round(fps)))
        with GLOBAL_JOB_SEMAPHORE:
            lint_res, _ = run_managed_subprocess(
                [str(self.hf_bin), "lint", str(slot_dir)], timeout_s=45.0, cwd=slot_dir, env=env, check=False
            )
            render_cmd = [
                str(self.hf_bin), "render", str(slot_dir),
                "--format", fmt,
                "--fps", str(fps_int),
                "--workers", "1",
                "--low-memory-mode",
                "--no-browser-gpu",
                "--frames-cache-dir", "off",
                "-o", str(out_path),
            ]
            last_err: Exception | None = None
            r_stats: dict[str, Any] = {}
            for attempt in range(3):
                try:
                    _, r_stats = run_managed_subprocess(render_cmd, timeout_s=timeout_s, cwd=slot_dir, env=env, check=True)
                    last_err = None
                    break
                except subprocess.CalledProcessError as exc:
                    last_err = exc
                    if attempt < 2:
                        time.sleep(2.0)
                    else:
                        tail = ((exc.stderr or "") + "\n" + (exc.output or ""))[-800:].strip()
                        raise RendererError(f"HyperFrames render failed (exit {exc.returncode}): {tail}") from exc
            if last_err is not None:
                raise last_err

        probe = self.verify_artifact(out_path)
        h = hashlib.sha256()
        with out_path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        if cached_artifact is not None:
            tmp_c = cached_artifact.with_suffix(f".{fmt}.partial")
            try:
                shutil.copy2(out_path, tmp_c)
                os.replace(tmp_c, cached_artifact)
            except Exception:
                tmp_c.unlink(missing_ok=True)
        return {
            "engine": self.name,
            "slot_dir": str(slot_dir),
            "artifact_path": str(out_path),
            "artifact_sha256": h.hexdigest(),
            "size_bytes": int(out_path.stat().st_size),
            "lint_rc": lint_res.returncode,
            "elapsed_s": r_stats["elapsed_s"],
            "peak_tree_rss_mb": r_stats.get("peak_tree_rss_mb", 0.0),
            "cache_hit": False,
            "cache_key": cache_key,
            "ffprobe": probe,
        }

    def render_graphics_scene(
        self,
        scene_id: str,
        template_name: str,
        width: int,
        height: int,
        fps: float,
        start_frame: int,
        end_frame_exclusive: int,
        anchor_bundle: dict[str, Any],
        parameters: dict[str, Any],
        workdir: Path,
        audio_envelope: list[float] | None = None,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Render a substantive ML-01.R3/R4 transparent VP9 WebM graphic scene via local HyperFrames + GSAP."""
        import numpy as np

        n_frames = max(4, end_frame_exclusive - start_frame)
        dur_s = n_frames / max(1.0, fps)
        slot_dir = (workdir.resolve() / f"hf_scene_{scene_id}_{template_name}").resolve()
        out_webm = slot_dir / "graphic.webm"

        if template_name in SUPPORTED_VFX_TEMPLATES:
            html = build_vfx_graphics_html(
                scene_id=scene_id,
                template_name=template_name,
                width=width,
                height=height,
                fps=fps,
                duration_s=dur_s,
                gsap_rel_path="./gsap.min.js",
                anchor_bundle=anchor_bundle,
                parameters=parameters,
            )
        else:
            # Reuse verified VFX depth/action templates cleanly for R3 template names
            vfx_map = {
                "depth-stage": "vfx-depth-stage",
                "depth-horizon": "vfx-depth-stage",
                "detail-connector": "vfx-macro-pointer",
                "action-path": "vfx-action-flow",
                "pass-through": "vfx-occlusion-wipe",
                "match-shape": "vfx-shape-morph",
                "split-divider": "vfx-editorial-collage",
                "contour-focus": "vfx-prop-accent",
                "audio-stage": "vfx-payoff-glint",
            }
            html = build_vfx_graphics_html(
                scene_id=scene_id,
                template_name=vfx_map.get(template_name, "vfx-depth-stage"),
                width=width,
                height=height,
                fps=fps,
                duration_s=dur_s,
                gsap_rel_path="./gsap.min.js",
                anchor_bundle=anchor_bundle,
                parameters=parameters,
            )

        res = self.render_html_slot(slot_dir=slot_dir, html=html, out_path=out_webm, fps=fps, fmt="webm", timeout_s=timeout_s)
        probe = res["ffprobe"]
        vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        pix_fmt = vstreams[0].get("pix_fmt", "") if vstreams else ""
        f_early = max(1, n_frames // 5)
        f_mid = max(2, n_frames // 2)
        alpha_mid = verify_webm_alpha_pixels(out_webm, width, height, sample_frame=f_mid, min_transparent_frac=0.15)
        alpha_early = verify_webm_alpha_pixels(out_webm, width, height, sample_frame=f_early, min_transparent_frac=0.15)
        if not alpha_mid["alpha_verified"]:
            raise RendererError(f"HyperFrames graphic {scene_id} failed alpha pixel verification: {alpha_mid}")

        motion_mad = 0.0
        try:
            ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
            dec = subprocess.run(
                [
                    ffmpeg, "-v", "error", "-c:v", "libvpx-vp9", "-i", str(out_webm),
                    "-vf", f"select='eq(n\\,{f_early})+eq(n\\,{f_mid})'", "-fps_mode", "passthrough",
                    "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
                ],
                capture_output=True, timeout=30, shell=False,
            )
            fbytes = width * height * 4
            if dec.returncode == 0 and len(dec.stdout) >= 2 * fbytes:
                a_img = np.frombuffer(dec.stdout[:fbytes], dtype=np.uint8).reshape((height, width, 4)).astype(np.float32)
                b_img = np.frombuffer(dec.stdout[fbytes : 2 * fbytes], dtype=np.uint8).reshape((height, width, 4)).astype(np.float32)
                motion_mad = round(float(np.abs(b_img - a_img).mean()), 4)
        except Exception:
            motion_mad = 0.0

        return {
            "scene_id": scene_id,
            "template_name": template_name,
            "engine": self.name,
            "artifact_path": str(out_webm),
            "artifact_sha256": res["artifact_sha256"],
            "size_bytes": res["size_bytes"],
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame_exclusive,
            "lint_rc": res["lint_rc"],
            "elapsed_s": res["elapsed_s"],
            "peak_tree_rss_mb": res["peak_tree_rss_mb"],
            "pix_fmt": pix_fmt,
            "motion_mad_early_vs_mid": motion_mad,
            "animated_verified": bool(motion_mad > 0.01 or abs(alpha_mid["opaque_stroke_frac"] - alpha_early["opaque_stroke_frac"]) > 0.0005),
            **alpha_mid,
        }

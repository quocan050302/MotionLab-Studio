"""Phase E3 — Local Offline Remotion Renderer Adapter.

Uses pinned local `renderers/remotion/node_modules/.bin/remotion` and an already-installed
local Chrome/Edge browser (`--browser-executable`), with zero `npx`, zero network downloads,
and explicit Remotion license governance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from motionlab.config import PROJECT_ROOT, atomic_write_json
from motionlab.renderers.base import BaseRenderer, RendererError, find_local_browser, probe_artifact_ffprobe
from motionlab.resource import GLOBAL_JOB_SEMAPHORE, ResourceBudget, run_managed_subprocess, run_subprocess_guarded
from motionlab.schemas import AnalysisResult, EditPlan, EffectEvent, EffectSpec, MediaInfo

REMOTION_SUPPORTED_EFFECTS = {
    "edge_lines",
    "foreground_frame_accent",
    "ring_accent",
    "directional_shapes",
    "ambient_particles",
    "waveform_accent",
    "pulse_border",
}

SUPPORTED_CREATIVE_RECIPES = {
    "RCP01", "RCP02", "RCP03", "RCP04", "RCP05", "RCP06", "RCP07", "RCP08", "RCP09",
    "VFX01", "VFX02", "VFX03", "VFX04", "VFX05", "VFX06",
    "VFX07", "VFX08", "VFX09", "VFX10", "VFX11", "VFX12",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

API_KEY_ENV_NAMES = (
    "ELEVENLABS_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "REPLICATE_API_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def _sanitized_offline_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in API_KEY_ENV_NAMES:
        env.pop(key, None)
    env["REMOTION_DISABLE_TELEMETRY"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["CI"] = "1"
    return env


class RemotionRenderer(BaseRenderer):
    name = "remotion"

    def __init__(self, license_allowed: bool = True, project_dir: Path | None = None) -> None:
        self.license_allowed = license_allowed
        self.project_dir = (project_dir or (PROJECT_ROOT / "renderers" / "remotion")).resolve()
        self.entry_point = self.project_dir / "src" / "index.ts"
        bin_name = "remotion.cmd" if os.name == "nt" else "remotion"
        self.remotion_bin = self.project_dir / "node_modules" / ".bin" / bin_name
        self.browser_exe = find_local_browser()

    def probe_capability(self) -> dict[str, Any]:
        license_note = (
            "Remotion License: Free for individuals, non-profits, evaluators, and for-profit teams <= 3 people; "
            "Company License required for for-profit organizations >= 4 people (https://www.remotion.dev/license)."
        )
        if not self.license_allowed:
            return {
                "available": False,
                "reason": "disabled_by_license_or_flag",
                "license": license_note,
            }
        if not self.remotion_bin.exists():
            return {
                "available": False,
                "reason": f"local_remotion_cli_missing:{self.remotion_bin}",
                "license": license_note,
            }
        if not self.entry_point.exists():
            return {
                "available": False,
                "reason": f"remotion_entry_missing:{self.entry_point}",
                "license": license_note,
            }
        if not self.browser_exe:
            return {
                "available": False,
                "reason": "local_chrome_or_edge_not_found_offline",
                "license": license_note,
            }
        pkg_json = self.project_dir / "package.json"
        version = "unknown"
        if pkg_json.exists():
            try:
                version = json.loads(pkg_json.read_text(encoding="utf-8")).get("dependencies", {}).get("remotion", "unknown")
            except Exception:
                pass
        return {
            "available": True,
            "version": version,
            "executable": str(self.remotion_bin),
            "browser_executable": self.browser_exe,
            "supported_effects": sorted(REMOTION_SUPPORTED_EFFECTS),
            "license": license_note,
            "reason": "ok",
        }

    def supports(self, effect_spec: EffectSpec, media: MediaInfo, capabilities: dict[str, Any]) -> bool:
        cap = capabilities.get("remotion") or {}
        return bool(cap.get("available")) and effect_spec.id in REMOTION_SUPPORTED_EFFECTS

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float | None = None) -> dict[str, Any]:
        info = probe_artifact_ffprobe(artifact_path)
        v_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not v_streams:
            raise RendererError(f"Remotion artifact has no video stream: {artifact_path}")
        pix_fmt = str(v_streams[0].get("pix_fmt") or "")
        tags = {str(k).lower(): str(v) for k, v in (v_streams[0].get("tags") or {}).items()}
        has_alpha = ("a" in pix_fmt.lower()) or (tags.get("alpha_mode") == "1")
        if not has_alpha:
            raise RendererError(
                f"Remotion WebM overlay did not preserve alpha channel (pix_fmt={pix_fmt}, tags={tags})"
            )
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
        """Render a single transparent WebM VP9 overlay clip using pinned local Remotion CLI."""
        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"Remotion unavailable: {cap.get('reason')}")

        t0 = time.monotonic()
        work_dir.mkdir(parents=True, exist_ok=True)
        out_webm.parent.mkdir(parents=True, exist_ok=True)

        dur_frames = max(2, event.end_frame_exclusive - event.start_frame)
        fps_int = max(1, int(round(plan.fps_num / plan.fps_den)))

        # Build exact per-frame audio envelope slice for audio-reactive overlays
        audio_lookup = {pt.frame: pt.rms_norm for pt in analysis.audio_envelope}
        env_slice: list[float] = []
        for rel_f in range(dur_frames):
            abs_f = event.start_frame + rel_f
            val = audio_lookup.get(abs_f)
            if val is None:
                for d in range(1, 5):
                    if (abs_f - d) in audio_lookup:
                        val = audio_lookup[abs_f - d]
                        break
                    if (abs_f + d) in audio_lookup:
                        val = audio_lookup[abs_f + d]
                        break
            env_slice.append(round(float(val or 0.0), 4))

        props = {
            "effectId": event.effect_id,
            "width": plan.width,
            "height": plan.height,
            "fps": fps_int,
            "durationInFrames": dur_frames,
            "intensity": float(event.parameters.get("intensity", 0.75)),
            "opacity": float(event.parameters.get("opacity", 0.65)),
            "marginRatio": float(event.parameters.get("margin_ratio", 0.045)),
            "seed": int(plan.seed),
            "regionCx": float(event.region.get("cx", 0.84)),
            "regionCy": float(event.region.get("cy", 0.86)),
            "barCount": int(event.parameters.get("bar_count", 16)),
            "particleCount": int(event.parameters.get("particle_count", 12)),
            "strokePx": int(event.parameters.get("stroke_px", 3)),
            "audioEnvelope": env_slice,
        }
        props_file = work_dir / f"remotion_props_{event.id}.json"
        atomic_write_json(props_file, props)

        cmd = [
            str(self.remotion_bin),
            "render",
            str(self.entry_point),
            "MotionLabOverlay",
            str(out_webm),
            f"--props={props_file}",
            "--codec=vp9",
            "--pixel-format=yuva420p",
            "--image-format=png",
            "--concurrency=1",
            "--log=error",
            f"--browser-executable={self.browser_exe}",
        ]
        res = run_subprocess_guarded(
            cmd,
            budget=budget,
            cwd=self.project_dir,
            env=_sanitized_offline_env(),
            timeout_s=min(240, budget.config.subprocess_timeout_s if budget else 240),
            check=False,
        )
        if res.returncode != 0:
            raise RendererError(
                f"Remotion render failed for {event.id} (exit {res.returncode}): {(res.stderr or res.stdout).strip()[:600]}"
            )
        info = self.verify_artifact(out_webm)
        return {
            "engine": "remotion",
            "artifact_path": str(out_webm),
            "elapsed_s": round(time.monotonic() - t0, 4),
            "ffprobe": info,
        }

    def supports_composition(self, recipe_id: str, capabilities: dict[str, Any]) -> bool:
        return (
            (recipe_id in SUPPORTED_CREATIVE_RECIPES or recipe_id.startswith("GEN_"))
            and bool((capabilities.get("remotion") or {}).get("available"))
        )

    def _slice_source_mezzanine(
        self,
        source_mp4: Path,
        start_frame: int,
        end_frame_exclusive: int,
        fps_str: str,
        out_mp4: Path,
        timeout_s: float = 90.0,
    ) -> Path:
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        n_frames = max(2, end_frame_exclusive - start_frame)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source_mp4),
            "-vf", f"select='between(n\\,{start_frame}\\,{end_frame_exclusive - 1})',setpts=N/({fps_str})/TB",
            "-frames:v", str(n_frames),
            "-r", fps_str,
            "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-g", "12", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart",
            str(out_mp4),
        ]
        with GLOBAL_JOB_SEMAPHORE:
            run_managed_subprocess(cmd, timeout_s=timeout_s, check=True)
        return out_mp4

    def render_scene_ablation_still(
        self,
        scene: Any,
        media: MediaInfo,
        scene_slot_dir: Path,
        sample_frame: int,
        exclude_layer_role: str | None,
        out_png: Path,
        composition_id: str = "MotionLabCreativeScene",
        timeout_s: float = 120.0,
    ) -> Path:
        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"Remotion unavailable: {cap.get('reason')}")
        public_dir = scene_slot_dir / "public"
        props_base = json.loads((scene_slot_dir / "props.json").read_text(encoding="utf-8"))
        props_ab = {**props_base, "excludeLayerRole": exclude_layer_role}
        ab_props_p = scene_slot_dir / f"props_ablate_{exclude_layer_role or 'full'}.json"
        atomic_write_json(ab_props_p, props_ab)

        rem_tmp = scene_slot_dir / "tmp"
        rem_tmp.mkdir(parents=True, exist_ok=True)
        env = _sanitized_offline_env()
        env["REMOTION_TMP"] = str(rem_tmp)
        env["TMPDIR"] = str(rem_tmp)
        env["TEMP"] = str(rem_tmp)
        env["TMP"] = str(rem_tmp)

        cmd = [
            str(self.remotion_bin), "still",
            str(self.entry_point),
            composition_id,
            str(out_png),
            f"--frame={max(0, int(sample_frame))}",
            f"--props={ab_props_p}",
            f"--public-dir={public_dir}",
            f"--browser-executable={self.browser_exe}",
            "--image-format=png",
            "--log=error",
        ]
        with GLOBAL_JOB_SEMAPHORE:
            run_managed_subprocess(cmd, timeout_s=timeout_s, cwd=self.project_dir, env=env, check=True)
        return out_png

    def render_scene_composition(
        self,
        scene: Any,
        media: MediaInfo,
        workdir: Path,
        matte_mp4: Path | None = None,
        prop_matte_mp4: Path | None = None,
        hf_graphic_webm: Path | None = None,
        exclude_layer_role: str | None = None,
        timeout_s: float = 240.0,
    ) -> dict[str, Any]:
        """Render a full-frame multi-layer creative scene via pinned local Remotion CLI."""
        from motionlab.composition.recipes import get_recipe
        from motionlab.composition.vfx import get_vfx_recipe, is_vfx_recipe
        from motionlab.vision.segmentation import build_foreground_alpha_webm

        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"Remotion unavailable: {cap.get('reason')}")

        spec = get_vfx_recipe(scene.recipe_id) if is_vfx_recipe(scene.recipe_id) else get_recipe(scene.recipe_id)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        tag = f"rem_scene_{scene.scene_id}" + (f"_ablate_{exclude_layer_role}" if exclude_layer_role else "")
        slot_dir = (workdir.resolve() / tag).resolve()
        public_dir = slot_dir / "public"
        public_dir.mkdir(parents=True, exist_ok=True)

        dur_frames = max(2, scene.end_frame_exclusive - scene.start_frame)
        fps_int = max(1, int(round(media.fps_float)))
        fps_str = media.fps.to_ffmpeg()

        src_slice = public_dir / "scene_source.mp4"
        if not src_slice.is_file():
            self._slice_source_mezzanine(
                Path(media.source_path), scene.start_frame, scene.end_frame_exclusive, fps_str, src_slice
            )

        fg_alpha_name: str | None = None
        if scene.requires_matte and matte_mp4 and matte_mp4.is_file():
            fg_alpha_p = public_dir / "fg_alpha.webm"
            if not fg_alpha_p.is_file():
                build_foreground_alpha_webm(
                    Path(media.source_path), matte_mp4, scene.start_frame, scene.end_frame_exclusive,
                    fps_str, fg_alpha_p, ffmpeg_bin=ffmpeg_bin,
                )
            fg_alpha_name = fg_alpha_p.name

        prop_alpha_name: str | None = None
        if prop_matte_mp4 and prop_matte_mp4.is_file():
            prop_alpha_p = public_dir / "prop_alpha.webm"
            if not prop_alpha_p.is_file():
                build_foreground_alpha_webm(
                    Path(media.source_path), prop_matte_mp4, scene.start_frame, scene.end_frame_exclusive,
                    fps_str, prop_alpha_p, ffmpeg_bin=ffmpeg_bin,
                )
            prop_alpha_name = prop_alpha_p.name

        hf_name: str | None = None
        if hf_graphic_webm and hf_graphic_webm.is_file():
            dst_hf = public_dir / "hf_graphic.webm"
            shutil.copy2(hf_graphic_webm, dst_hf)
            hf_name = dst_hf.name

        props = {
            "sceneId": scene.scene_id,
            "recipeId": scene.recipe_id,
            "sceneKind": spec.remotion_scene_kind,
            "width": media.display_width,
            "height": media.display_height,
            "fps": fps_int,
            "durationInFrames": dur_frames,
            "entryFrames": scene.entry_frames,
            "exitFrames": scene.exit_frames,
            "sourceFile": src_slice.name,
            "fgAlphaFile": fg_alpha_name,
            "propAlphaFile": prop_alpha_name,
            "hfGraphicFile": hf_name,
            "anchors": scene.anchor_bundle,
            "parameters": scene.parameters,
            "excludeLayerRole": exclude_layer_role,
        }
        props_path = slot_dir / "props.json"
        atomic_write_json(props_path, props)

        out_mp4 = slot_dir / "scene.mp4"
        rem_tmp = slot_dir / "tmp"
        rem_tmp.mkdir(parents=True, exist_ok=True)
        env = _sanitized_offline_env()
        env["REMOTION_TMP"] = str(rem_tmp)
        env["TMPDIR"] = str(rem_tmp)
        env["TEMP"] = str(rem_tmp)
        env["TMP"] = str(rem_tmp)

        cmd = [
            str(self.remotion_bin), "render",
            str(self.entry_point),
            "MotionLabCreativeScene",
            str(out_mp4),
            f"--props={props_path}",
            f"--public-dir={public_dir}",
            f"--browser-executable={self.browser_exe}",
            "--codec=h264",
            "--crf=16",
            "--pixel-format=yuv420p",
            "--muted",
            "--concurrency=1",
            "--log=error",
        ]
        with GLOBAL_JOB_SEMAPHORE:
            proc = None
            stats: dict[str, Any] = {}
            for attempt in range(2):
                try:
                    proc, stats = run_managed_subprocess(cmd, timeout_s=timeout_s, cwd=self.project_dir, env=env, check=True)
                    break
                except subprocess.CalledProcessError as exc:
                    if attempt == 0:
                        time.sleep(1.5)
                    else:
                        tail = ((exc.stderr or "") + "\n" + (exc.output or ""))[-800:].strip()
                        raise RendererError(f"Remotion scene {scene.scene_id} failed (exit {exc.returncode}): {tail}") from exc
            assert proc is not None

        probe = probe_artifact_ffprobe(out_mp4)
        vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        if not vstreams:
            raise RendererError(f"Remotion scene {scene.scene_id} produced 0 video streams: {out_mp4}")
        vs = vstreams[0]
        actual_w, actual_h = int(vs.get("width", 0)), int(vs.get("height", 0))
        actual_frames = int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else dur_frames
        if actual_w != media.display_width or actual_h != media.display_height:
            raise RendererError(f"Remotion scene {scene.scene_id} dimension mismatch: {actual_w}x{actual_h}")
        if actual_frames != dur_frames:
            raise RendererError(f"Remotion scene {scene.scene_id} frame count mismatch: {actual_frames} vs {dur_frames}")

        return {
            "scene_id": scene.scene_id,
            "recipe_id": scene.recipe_id,
            "scene_kind": spec.remotion_scene_kind,
            "engine": self.name,
            "slot_dir": str(slot_dir),
            "artifact_path": str(out_mp4),
            "artifact_sha256": _sha256_file(out_mp4),
            "size_bytes": int(out_mp4.stat().st_size),
            "start_frame": scene.start_frame,
            "end_frame_exclusive": scene.end_frame_exclusive,
            "frame_count": actual_frames,
            "dimensions": f"{actual_w}x{actual_h}",
            "pix_fmt": str(vs.get("pix_fmt", "")),
            "used_matte_alpha": fg_alpha_name is not None,
            "used_prop_alpha": prop_alpha_name is not None,
            "used_hyperframes_graphics": hf_name is not None,
            "exclude_layer_role": exclude_layer_role,
            "elapsed_s": stats["elapsed_s"],
            "peak_tree_rss_mb": stats.get("peak_tree_rss_mb", 0.0),
            "cli_stdout_tail": (proc.stdout or "")[-400:].strip(),
        }

    def render_custom_composition(
        self,
        composition_id: str,
        props: dict[str, Any],
        out_path: Path,
        slot_dir: Path,
        public_dir: Path | None = None,
        codec: str = "h264",
        pixel_format: str = "yuv420p",
        timeout_s: float = 240.0,
    ) -> dict[str, Any]:
        """Generic offline Remotion render helper for R5 imported/generative compositions & benchmarks."""
        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"Remotion unavailable: {cap.get('reason')}")
        slot_dir = slot_dir.resolve()
        slot_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        props_path = slot_dir / "props.json"
        atomic_write_json(props_path, props)
        rem_tmp = (PROJECT_ROOT / ".cache" / "remotion_tmp").resolve()
        rem_tmp.mkdir(parents=True, exist_ok=True)
        env = _sanitized_offline_env()
        env["REMOTION_TMP"] = str(rem_tmp)
        env["TMPDIR"] = str(rem_tmp)
        env["TEMP"] = str(rem_tmp)
        env["TMP"] = str(rem_tmp)

        cmd = [
            str(self.remotion_bin), "render",
            str(self.entry_point),
            composition_id,
            str(out_path),
            f"--props={props_path}",
            f"--browser-executable={self.browser_exe}",
            f"--codec={codec}",
            f"--pixel-format={pixel_format}",
            "--bundle-cache",
            "--muted",
            "--concurrency=1",
            "--log=error",
        ]
        if codec == "h264":
            cmd.append("--crf=16")
        if codec == "vp9":
            cmd.append("--image-format=png")
        if public_dir is not None:
            cmd.append(f"--public-dir={public_dir.resolve()}")

        with GLOBAL_JOB_SEMAPHORE:
            proc = None
            stats: dict[str, Any] = {}
            for attempt in range(2):
                try:
                    proc, stats = run_managed_subprocess(cmd, timeout_s=timeout_s, cwd=self.project_dir, env=env, check=True)
                    break
                except subprocess.CalledProcessError as exc:
                    if attempt == 0:
                        time.sleep(1.5)
                    else:
                        tail = ((exc.stderr or "") + "\n" + (exc.output or ""))[-800:].strip()
                        raise RendererError(f"Remotion {composition_id} failed (exit {exc.returncode}): {tail}") from exc
            assert proc is not None
        probe = probe_artifact_ffprobe(out_path)
        return {
            "composition_id": composition_id,
            "engine": self.name,
            "artifact_path": str(out_path),
            "artifact_sha256": _sha256_file(out_path),
            "size_bytes": int(out_path.stat().st_size),
            "elapsed_s": stats["elapsed_s"],
            "peak_tree_rss_mb": stats.get("peak_tree_rss_mb", 0.0),
            "ffprobe": probe,
            "stdout_tail": (proc.stdout or "")[-300:].strip(),
        }

    def compute_code_hash(self) -> str:
        """Compute deterministic content hash over all Remotion TSX/TS sources + package.json/lock."""
        h = hashlib.sha256()
        src_root = self.project_dir / "src"
        files: list[Path] = []
        if src_root.is_dir():
            files.extend(sorted(p for p in src_root.rglob("*") if p.is_file() and p.suffix in (".ts", ".tsx", ".css", ".json")))
        for extra in ("package.json", "package-lock.json", "tsconfig.json"):
            ep = self.project_dir / extra
            if ep.is_file():
                files.append(ep)
        for fp in files:
            rel = str(fp.relative_to(self.project_dir)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            h.update(fp.read_bytes())
        return h.hexdigest()[:20]

    def render_batch_jobs(
        self,
        *,
        jobs: list[dict[str, Any]],
        public_mounts: list[dict[str, str]],
        work_dir: Path,
        bundle_cache_root: Path | None = None,
        timeout_s: float = 600.0,
    ) -> dict[str, Any]:
        """Render multiple Remotion scenes & ablation stills in ONE persistent Chromium browser + ONE cached bundle.

        Falls back fail-closed if `batch_render.mjs` or local Node is missing.
        """
        cap = self.probe_capability()
        if not cap.get("available"):
            raise RendererError(f"Remotion unavailable: {cap.get('reason')}")
        node_bin = shutil.which("node")
        batch_script = self.project_dir / "scripts" / "batch_render.mjs"
        if not node_bin or not batch_script.is_file():
            raise RendererError(f"Remotion batch worker unavailable (node={node_bin}, script={batch_script})")

        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        code_hash = self.compute_code_hash()
        b_root = (bundle_cache_root or (PROJECT_ROOT / ".cache" / "motionlab_r5" / "remotion_bundles")).resolve()
        bundle_dir = b_root / f"bundle_{code_hash}"
        serve_dir = work_dir / "remotion_serve"
        result_path = work_dir / "batch_result.json"
        manifest_path = work_dir / "batch_manifest.json"

        atomic_write_json(
            manifest_path,
            {
                "entryPoint": str(self.entry_point),
                "bundleCacheDir": str(bundle_dir),
                "codeHash": code_hash,
                "serveDir": str(serve_dir),
                "browserExecutable": self.browser_exe,
                "publicMounts": public_mounts,
                "jobs": jobs,
                "resultPath": str(result_path),
            },
        )

        rem_tmp = (PROJECT_ROOT / ".cache" / "remotion_tmp").resolve()
        rem_tmp.mkdir(parents=True, exist_ok=True)
        env = _sanitized_offline_env()
        env["REMOTION_TMP"] = str(rem_tmp)
        env["TMPDIR"] = str(rem_tmp)
        env["TEMP"] = str(rem_tmp)
        env["TMP"] = str(rem_tmp)

        cmd = [node_bin, str(batch_script), f"--manifest={manifest_path}"]
        with GLOBAL_JOB_SEMAPHORE:
            proc, stats = run_managed_subprocess(cmd, timeout_s=timeout_s, cwd=self.project_dir, env=env, check=False)
        if proc.returncode != 0 or not result_path.is_file():
            tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-1000:].strip()
            raise RendererError(f"Remotion batch worker failed (exit {proc.returncode}): {tail}")

        summary = json.loads(result_path.read_text(encoding="utf-8"))
        summary["peak_tree_rss_mb"] = stats.get("peak_tree_rss_mb", 0.0)
        return summary

    def render_v4_scene_overlay(
        self,
        *,
        composition_id: str = "SceneGraphV4Composition",
        props: dict[str, Any],
        out_path: Path,
        extra_assets: dict[str, Path] | None = None,
        work_dir: Path,
        timeout_s: float = 240.0,
    ) -> dict[str, Any]:
        """Render a transparent VP9 WebM overlay via local offline Remotion CLI for SceneGraph V4."""
        cap = self.probe_capability()
        if not cap.get("available"):
            return {"rendered": False, "reason": cap.get("reason")}
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        props_file = work_dir / "v4_props.json"
        atomic_write_json(props_file, props)

        # Stage local extra assets into Remotion public/ and clean up cleanly after render
        pub_dir = self.project_dir / "public"
        pub_dir.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        try:
            for k, src in (extra_assets or {}).items():
                dst = pub_dir / str(k)
                shutil.copy2(Path(src), dst)
                staged.append(dst)

            rem_tmp = (PROJECT_ROOT / ".cache" / "remotion_tmp").resolve()
            rem_tmp.mkdir(parents=True, exist_ok=True)
            env = _sanitized_offline_env()
            env["REMOTION_TMP"] = str(rem_tmp)
            env["TMPDIR"] = str(rem_tmp)
            env["TEMP"] = str(rem_tmp)
            env["TMP"] = str(rem_tmp)

            cmd = [
                str(self.remotion_bin),
                "render",
                "src/index.ts",
                composition_id,
                str(out_path),
                "--codec=vp9",
                "--image-format=png",
                "--pixel-format=yuva420p",
                "--concurrency=2",
                f"--props={props_file}",
            ]
            if self.browser_exe:
                cmd.append(f"--browser-executable={self.browser_exe}")
            t0 = time.perf_counter()
            with GLOBAL_JOB_SEMAPHORE:
                proc, stats = run_managed_subprocess(cmd, timeout_s=timeout_s, cwd=self.project_dir, env=env, check=False)
            elapsed_s = round(time.perf_counter() - t0, 3)
            if proc.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
                tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-600:].strip()
                return {"rendered": False, "exit_code": proc.returncode, "error": tail}
            return {
                "rendered": True,
                "output_path": str(out_path),
                "sha256": _sha256_file(out_path),
                "bytes": int(out_path.stat().st_size),
                "elapsed_s": elapsed_s,
                "peak_tree_rss_mb": stats.get("peak_tree_rss_mb", 0.0),
            }
        finally:
            for sp in staged:
                sp.unlink(missing_ok=True)

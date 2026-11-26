"""Phase E2, E5, E6, E7: Full-length non-destructive FFmpeg renderer with hard audio-master lock."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.config import MotionLabConfig
from motionlab.effects.engine_frames import apply_frame_effects
from motionlab.renderers.base import BaseRenderer
from motionlab.resource import ResourceBudget, ResourceTracker, run_managed_subprocess, terminate_process_tree
from motionlab.schemas import AnalysisResult, EditPlan, EffectEvent, EffectSpec, MediaInfo, RenderResult

TONEMAP_HDR_TO_SDR = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
    "format=rgb24"
)


def _build_decode_vf(
    media: MediaInfo,
    plan: EditPlan,
    config: MotionLabConfig,
    analysis: AnalysisResult | None,
) -> tuple[str, list[str]]:
    """Build deterministic FFmpeg input framing filtergraph (preserves aspect or safe 9:16 pad/smart-crop)."""
    w, h = plan.width, plan.height
    notes: list[str] = []
    parts: list[str] = []

    if media.is_hdr and config.allow_hdr_tonemap:
        parts.append(TONEMAP_HDR_TO_SDR)
        notes.append("explicit_hdr_to_sdr_tonemap_applied")

    if media.vfr and config.allow_vfr_cfr_normalize:
        parts.append(f"fps={plan.fps.to_ffmpeg()}")
        notes.append(f"vfr_normalized_to_rational_cfr:{plan.fps.to_ffmpeg()}")

    if plan.aspect_policy == "preserve":
        if media.display_width != w or media.display_height != h:
            parts.append(f"scale={w}:{h}:flags=lanczos")
        return ",".join(parts), notes

    if plan.aspect_policy == "smart_crop_9x16":
        # Compute stable horizontal crop center from detected subject regions
        cx = 0.5
        if analysis and analysis.subject_regions:
            xs = [s.x + s.w * 0.5 for s in analysis.subject_regions if s.confidence >= 0.6 and not s.ambiguous]
            if xs:
                cx = max(0.25, min(0.75, sum(xs) / len(xs)))
        # Scale to cover target height, then crop width centered at cx
        parts.append(f"scale=-2:{h}:flags=lanczos")
        parts.append(f"crop={w}:{h}:max(0\\,min(iw-{w}\\,iw*{cx:.4f}-{w}/2)):0")
        notes.append(f"smart_crop_9x16_center_x={cx:.3f}")
        return ",".join(parts), notes

    if plan.aspect_policy == "pad_solid_9x16":
        parts.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos")
        parts.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=#101218")
        notes.append("pad_solid_9x16")
        return ",".join(parts), notes

    # Default pad_blur_9x16 uses filter_complex handled separately
    notes.append("pad_blur_9x16")
    return "__PAD_BLUR_9X16__", notes


def _build_decode_cmd(
    ffmpeg_bin: str,
    media: MediaInfo,
    plan: EditPlan,
    config: MotionLabConfig,
    analysis: AnalysisResult | None,
    threads: int,
) -> tuple[list[str], list[str]]:
    vf, notes = _build_decode_vf(media, plan, config, analysis)
    w, h = plan.width, plan.height

    base = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-threads", str(threads),
        "-i", media.path,
    ]
    if vf == "__PAD_BLUR_9X16__":
        pre = []
        if media.is_hdr and config.allow_hdr_tonemap:
            pre.append(TONEMAP_HDR_TO_SDR)
        if media.vfr and config.allow_vfr_cfr_normalize:
            pre.append(f"fps={plan.fps.to_ffmpeg()}")
        pre_chain = (",".join(pre) + ",") if pre else ""
        fc = (
            f"[0:v]{pre_chain}split=2[bg][fg];"
            f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase:flags=bilinear,"
            f"crop={w}:{h},gblur=sigma=24,eq=brightness=-0.12[bgb];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos[fgf];"
            f"[bgb][fgf]overlay=(W-w)/2:(H-h)/2,format=rgb24[outv]"
        )
        cmd = base + ["-filter_complex", fc, "-map", "[outv]", "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn", "-dn", "pipe:1"]
        return cmd, notes

    cmd = list(base)
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn", "-dn", "pipe:1"]
    return cmd, notes


def _select_video_encoder(ffmpeg_bin: str) -> tuple[str, list[str]]:
    """Prefer deterministic software libx264 for uniform cross-platform quality."""
    try:
        out = subprocess.run([ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=6, shell=False)
        if "libx264" in out.stdout:
            return "libx264", ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    except (OSError, subprocess.SubprocessError):
        pass
    return "mpeg4", ["-c:v", "mpeg4", "-pix_fmt", "yuv420p"]


class FFmpegRenderer(BaseRenderer):
    """Primary local frame-exact streaming renderer + lossless audio-master muxer."""

    name = "ffmpeg"

    def __init__(self, ffmpeg_bin: str | None = None, ffprobe_bin: str | None = None) -> None:
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_bin = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"

    def supports(
        self,
        spec: EffectSpec,
        media: MediaInfo,
        capabilities: dict[str, Any],
    ) -> bool:
        return "ffmpeg" in spec.supported_engines and bool(capabilities.get("ffmpeg", {}).get("available"))

    def prepare(self, plan: EditPlan, source: Path, workdir: Path) -> dict[str, Any]:
        workdir.mkdir(parents=True, exist_ok=True)
        enc_name, _ = _select_video_encoder(self.ffmpeg_bin)
        return {"workdir": str(workdir), "encoder": enc_name, "ffmpeg": self.ffmpeg_bin}

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float) -> dict[str, Any]:
        proc = subprocess.run(
            [
                self.ffprobe_bin, "-v", "error",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(artifact_path),
            ],
            capture_output=True, text=True, timeout=20, shell=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed on rendered artifact {artifact_path}: {proc.stderr}")
        return json.loads(proc.stdout)

    def render_control_encode(
        self,
        media: MediaInfo,
        plan: EditPlan,
        config: MotionLabConfig,
        analysis: AnalysisResult | None,
        out_path: Path,
        budget: ResourceBudget,
    ) -> Path:
        """Render an effect-free baseline control encode with identical layout & codec settings."""
        empty_plan = EditPlan(
            schema_version=plan.schema_version,
            planner_version=plan.planner_version,
            source_hash=plan.source_hash,
            config_hash=plan.config_hash,
            seed=plan.seed,
            profile=plan.profile,
            profile_selection_reason="control_no_effects",
            width=plan.width,
            height=plan.height,
            fps=plan.fps,
            duration_frames=plan.duration_frames,
            aspect_policy=plan.aspect_policy,
            audio_policy={**plan.audio_policy, "mode": "none", "has_source_audio": False},
            engine_capabilities=plan.engine_capabilities,
            events=[],
            status="NO_SAFE_EFFECTS",
        )
        self._stream_render_video_only(
            media=media,
            plan=empty_plan,
            config=config,
            analysis=analysis,
            video_only_path=out_path,
            budget=budget,
            external_overlays=[],
            tracker=None,
        )
        return out_path

    def _stream_render_video_only(
        self,
        media: MediaInfo,
        plan: EditPlan,
        config: MotionLabConfig,
        analysis: AnalysisResult | None,
        video_only_path: Path,
        budget: ResourceBudget,
        external_overlays: list[dict[str, Any]],
        tracker: ResourceTracker | None,
    ) -> dict[str, Any]:
        """Stream decoded RGB24 frames -> ring buffer -> apply_frame_effects -> encode H.264 video-only MP4."""
        w, h = plan.width, plan.height
        frame_bytes = w * h * 3
        fps_str = plan.fps.to_ffmpeg()
        threads = max(1, budget.ffmpeg_threads)

        decode_cmd, decode_notes = _build_decode_cmd(
            self.ffmpeg_bin, media, plan, config, analysis, threads=threads
        )
        enc_name, enc_args = _select_video_encoder(self.ffmpeg_bin)
        crf = "23" if config.quality == "preview" else "18"
        preset = "fast" if config.quality == "preview" else "medium"

        encode_cmd = [
            self.ffmpeg_bin, "-y",
            "-hide_banner", "-loglevel", "error",
            "-threads", str(threads),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", fps_str,
            "-i", "pipe:0",
            *enc_args,
        ]
        if enc_name == "libx264":
            encode_cmd += ["-preset", preset, "-crf", crf, "-g", str(max(15, int(round(plan.fps.as_float() * 2))))]
        encode_cmd += [
            "-an",
            "-movflags", "+faststart",
            str(video_only_path),
        ]

        # Build frame-aligned audio envelope array for G05/G06
        audio_env: np.ndarray | None = None
        if analysis and analysis.audio_envelope:
            audio_env = np.array([p.normalized_rms for p in analysis.audio_envelope], dtype=np.float32)

        # Exclude events handled by external overlay clips so they aren't applied twice
        ext_event_ids = {ov["event_id"] for ov in external_overlays}
        local_events = [ev for ev in plan.events if ev.id not in ext_event_ids]

        # Compute max lookahead radius needed by neighbor-frame transitions/echoes (bounded <= 16 frames!)
        lookahead = 0
        for ev in local_events:
            if ev.uses_neighbor_frames:
                half = (ev.end_frame_exclusive - ev.start_frame) // 2 + 1
                lookahead = max(lookahead, min(16, half))

        dec_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        enc_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        assert dec_proc.stdout is not None and enc_proc.stdin is not None

        ring: dict[int, np.ndarray] = {}
        read_idx = 0
        write_idx = 0
        eof = False
        t0 = time.monotonic()

        try:
            while True:
                # Fill lookahead window in bounded ring buffer
                while not eof and (read_idx <= write_idx + lookahead):
                    raw = dec_proc.stdout.read(frame_bytes)
                    if not raw or len(raw) < frame_bytes:
                        eof = True
                        break
                    frame_arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
                    ring[read_idx] = frame_arr
                    read_idx += 1

                if write_idx not in ring:
                    break

                cur = ring[write_idx]
                active = [
                    ev for ev in local_events
                    if ev.start_frame <= write_idx < ev.end_frame_exclusive
                ]
                if active:
                    out_frame = apply_frame_effects(
                        cur,
                        active,
                        write_idx,
                        ring,
                        audio_env,
                        plan.seed,
                    )
                else:
                    out_frame = cur

                enc_proc.stdin.write(out_frame.tobytes())

                # Evict frames older than bounded lookbehind (max 16 frames)
                evict_below = write_idx - max(16, lookahead + 2)
                for old_k in [k for k in ring if k < evict_below]:
                    del ring[old_k]

                write_idx += 1
                if tracker is not None and (write_idx % 60 == 0):
                    tracker.sample_self()
                if time.monotonic() - t0 > config.timeout_s:
                    raise TimeoutError(f"FFmpeg frame render timed out after {config.timeout_s}s")
        except Exception:
            terminate_process_tree(dec_proc.pid)
            terminate_process_tree(enc_proc.pid)
            raise
        finally:
            try:
                enc_proc.stdin.close()
            except OSError:
                pass
            try:
                dec_proc.stdout.close()
            except OSError:
                pass

        dec_err = dec_proc.stderr.read().decode("utf-8", errors="replace") if dec_proc.stderr else ""
        enc_err = enc_proc.stderr.read().decode("utf-8", errors="replace") if enc_proc.stderr else ""
        if dec_proc.stderr:
            dec_proc.stderr.close()
        if enc_proc.stderr:
            enc_proc.stderr.close()

        dec_rc = dec_proc.wait(timeout=30)
        enc_rc = enc_proc.wait(timeout=60)
        if dec_rc != 0:
            raise RuntimeError(f"FFmpeg decode stream failed (rc={dec_rc}): {dec_err[:500]}")
        if enc_rc != 0:
            raise RuntimeError(f"FFmpeg encode stream failed (rc={enc_rc}): {enc_err[:500]}")
        if write_idx == 0:
            raise RuntimeError("FFmpeg rendered 0 frames.")

        return {
            "frames_rendered": write_idx,
            "encoder": enc_name,
            "crf": crf,
            "preset": preset,
            "pix_fmt": "yuv420p",
            "decode_notes": decode_notes,
            "elapsed_s": round(time.monotonic() - t0, 4),
        }

    def _composite_external_overlays(
        self,
        base_video: Path,
        external_overlays: list[dict[str, Any]],
        out_video: Path,
        plan: EditPlan,
        config: MotionLabConfig,
        budget: ResourceBudget,
    ) -> dict[str, Any]:
        """Composite verified external engine overlay clips (HyperFrames/Remotion) with exact PTS shift."""
        inputs: list[str] = ["-i", str(base_video)]
        for ov in external_overlays:
            # Explicit libvpx-vp9 decoder when WebM alpha is supplied so alpha plane survives decode!
            ov_path = Path(ov["artifact_path"])
            if ov_path.suffix.lower() == ".webm":
                inputs += ["-c:v", "libvpx-vp9", "-i", str(ov_path)]
            else:
                inputs += ["-i", str(ov_path)]

        filter_parts: list[str] = []
        current = "[0:v]"
        for idx, ov in enumerate(external_overlays, start=1):
            t_start = plan.fps.frame_to_seconds(int(ov["start_frame"]))
            t_end = plan.fps.frame_to_seconds(int(ov["end_frame_exclusive"]))
            filter_parts.append(
                f"[{idx}:v]format=rgba,scale={plan.width}:{plan.height},"
                f"setpts=PTS-STARTPTS+{t_start:.6f}/TB[ov{idx}]"
            )
            nxt = f"[v{idx}]"
            filter_parts.append(
                f"{current}[ov{idx}]overlay=0:0:enable='between(t,{t_start:.6f},{t_end:.6f})':eof_action=pass{nxt}"
            )
            current = nxt

        enc_name, enc_args = _select_video_encoder(self.ffmpeg_bin)
        crf = "22" if config.quality == "preview" else "18"
        preset = "fast" if config.quality == "preview" else "medium"
        cmd = [
            self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-threads", str(max(1, budget.ffmpeg_threads)),
            *inputs,
            "-filter_complex", ";".join(filter_parts),
            "-map", current,
            *enc_args,
        ]
        if enc_name == "libx264":
            cmd += ["-preset", preset, "-crf", crf]
        cmd += ["-an", "-movflags", "+faststart", str(out_video)]
        _, stats = run_managed_subprocess(cmd, timeout_s=config.timeout_s)
        return stats

    def _mux_master_audio(
        self,
        video_only_path: Path,
        media: MediaInfo,
        plan: EditPlan,
        out_partial_path: Path,
        config: MotionLabConfig,
    ) -> tuple[dict[str, Any], list[str]]:
        """Phase E6 HARD LOCK: Mux master audio ONCE without filters, fades, retime, or -shortest."""
        warnings: list[str] = []
        audio_pol = plan.audio_policy
        has_audio = bool(audio_pol.get("has_source_audio")) and media.has_audio

        if not has_audio:
            cmd = [
                self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_only_path),
                "-map", "0:v:0",
                "-c:v", "copy",
                "-an",
                "-movflags", "+faststart",
                str(out_partial_path),
            ]
            _, stats = run_managed_subprocess(cmd, timeout_s=config.timeout_s)
            return {"audio_mux_mode": "none", "elapsed_s": stats["elapsed_s"]}, warnings

        selected_tracks: list[int] = list(audio_pol.get("selected_tracks") or [0])
        mode = str(audio_pol.get("mode", "stream_copy"))

        # Preserve relative start offset between video and audio if non-zero
        primary_a = media.audio_streams[selected_tracks[0]]
        rel_offset_s = primary_a.start_time_s - media.video_start_time_s

        cmd = [
            self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_only_path),
        ]
        if abs(rel_offset_s) > 0.005:
            cmd += ["-itsoffset", f"{rel_offset_s:.6f}"]
            warnings.append(f"Preserved non-zero A/V start offset ({rel_offset_s:+.4f}s) during final master mux.")
        cmd += [
            "-i", media.path,
            "-map", "0:v:0",
        ]
        for tr_idx in selected_tracks:
            cmd += ["-map", f"1:a:{tr_idx}"]

        cmd += ["-c:v", "copy"]
        if mode == "stream_copy":
            cmd += ["-c:a", "copy"]
        else:
            # Disclosed high-bitrate AAC transcode ONLY when container cannot copy raw PCM/unsupported audio codec
            cmd += ["-c:a", "aac", "-b:a", "256k"]
            warnings.append(
                f"Disclosed audio transcode (AAC 256k): source audio codec '{primary_a.codec_name}' "
                "is not directly MP4 stream-copy compatible; no audio filters/fades/retime applied."
            )

        # IMPORTANT: NEVER pass `-shortest` so neither audio nor video is silently truncated!
        cmd += ["-movflags", "+faststart", str(out_partial_path)]
        _, stats = run_managed_subprocess(cmd, timeout_s=config.timeout_s)
        return {
            "audio_mux_mode": mode,
            "selected_tracks": selected_tracks,
            "relative_offset_s": round(rel_offset_s, 6),
            "elapsed_s": stats["elapsed_s"],
        }, warnings

    def render_timeline(
        self,
        media: MediaInfo,
        plan: EditPlan,
        config: MotionLabConfig,
        analysis: AnalysisResult | None,
        workdir: Path,
        out_partial_path: Path,
        budget: ResourceBudget,
        external_overlays: list[dict[str, Any]] | None = None,
        fallback_history: list[dict[str, Any]] | None = None,
        tracker: ResourceTracker | None = None,
    ) -> RenderResult:
        """Render entire non-destructive timeline to `out_partial_path`."""
        self.prepare(plan, Path(media.path), workdir)
        overlays = external_overlays or []
        timings: dict[str, float] = {}
        warnings: list[str] = list(plan.warnings)

        t_vid = time.monotonic()
        video_only = workdir / "video_only.mp4"
        enc_info = self._stream_render_video_only(
            media=media,
            plan=plan,
            config=config,
            analysis=analysis,
            video_only_path=video_only,
            budget=budget,
            external_overlays=overlays,
            tracker=tracker,
        )
        timings["video_effects_render_s"] = round(time.monotonic() - t_vid, 4)

        final_video_track = video_only
        if overlays:
            t_ov = time.monotonic()
            composited = workdir / "video_composited.mp4"
            ov_stats = self._composite_external_overlays(
                video_only, overlays, composited, plan, config, budget
            )
            timings["external_overlay_composite_s"] = round(time.monotonic() - t_ov, 4)
            if tracker is not None:
                tracker.record_child_peak(ov_stats.get("peak_rss_mb", 0.0))
            final_video_track = composited

        t_mux = time.monotonic()
        mux_info, mux_warnings = self._mux_master_audio(
            final_video_track, media, plan, out_partial_path, config
        )
        timings["audio_master_mux_s"] = round(time.monotonic() - t_mux, 4)
        warnings.extend(mux_warnings)

        probe_info = self.verify_artifact(out_partial_path, media.duration_s)
        event_engine_map: dict[str, str] = {}
        ov_by_id = {ov["event_id"]: ov["engine"] for ov in overlays}
        for ev in plan.events:
            event_engine_map[ev.id] = ov_by_id.get(ev.id, "ffmpeg")

        engines_used = sorted(set(["ffmpeg"] + [ov["engine"] for ov in overlays]))
        return RenderResult(
            artifact_path=str(out_partial_path),
            engines_used=engines_used,
            event_engine_map=event_engine_map,
            timings_s=timings,
            ffprobe_info=probe_info,
            status="PASS",
            warnings=warnings,
            fallback_history=fallback_history or [],
            encoding_settings={**enc_info, **mux_info},
        )

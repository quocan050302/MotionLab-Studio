"""Phase E2/E5/E6 — Non-destructive Streaming FFmpeg Renderer & Hard-Locked Audio Muxer.

Renders all 40 ML-01 effects deterministically from integer frame indices using:
  - Bounded streaming decode pipe (never holds full video in RAM)
  - Small rolling frame buffer (<= 16 frames) only when neighbor-frame effects exist
  - Optional external alpha overlay compositing (Remotion / HyperFrames WebM VP9 yuva420p)
  - Final atomic-safe mux preserving source audio master via stream-copy without fades/loudnorm/retime
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.config import EditConfig
from motionlab.effects.implementations import EFFECT_FUNCTIONS, FrameRenderContext, apply_effect_frame
from motionlab.preflight import find_binary
from motionlab.renderers.base import BaseRenderer, RendererError, probe_artifact_ffprobe
from motionlab.resource import ResourceBudget, run_subprocess_guarded
from motionlab.schemas import AnalysisResult, EditPlan, EffectEvent, EffectSpec, MediaInfo

TONEMAP_VF = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv"
)

LAYER_ORDER = {"transition": 0, "transform": 1, "layout": 2, "grade": 3, "overlay": 4}


class FFmpegRenderer(BaseRenderer):
    name = "ffmpeg"

    def __init__(self, ffmpeg_bin: str | None = None, ffprobe_bin: str | None = None) -> None:
        self.ffmpeg_bin = ffmpeg_bin or find_binary("ffmpeg")
        self.ffprobe_bin = ffprobe_bin or find_binary("ffprobe")

    def probe_capability(self) -> dict[str, Any]:
        if not self.ffmpeg_bin or not self.ffprobe_bin:
            return {
                "available": False,
                "reason": "ffmpeg_or_ffprobe_not_found",
                "license": "GPL/LGPL static binary (system installed)",
            }
        try:
            ver_proc = subprocess.run(
                [self.ffmpeg_bin, "-version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            first_line = (ver_proc.stdout.splitlines() or ["unknown"])[0]
            enc_proc = subprocess.run(
                [self.ffmpeg_bin, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            has_x264 = "libx264" in enc_proc.stdout
            has_vp9 = "libvpx-vp9" in enc_proc.stdout
            has_aac = " aac " in enc_proc.stdout
            return {
                "available": has_x264,
                "version": first_line,
                "executable": self.ffmpeg_bin,
                "ffprobe": self.ffprobe_bin,
                "encoders": {"libx264": has_x264, "libvpx_vp9": has_vp9, "aac": has_aac},
                "supported_effect_count": len(EFFECT_FUNCTIONS),
                "license": "FFmpeg static build (GPLv3/LGPL; check local build banner)",
                "reason": "ok" if has_x264 else "libx264_missing",
            }
        except Exception as exc:
            return {"available": False, "reason": f"probe_error:{exc}"}

    def supports(self, effect_spec: EffectSpec, media: MediaInfo, capabilities: dict[str, Any]) -> bool:
        cap = capabilities.get("ffmpeg") or {}
        return bool(cap.get("available")) and effect_spec.id in EFFECT_FUNCTIONS

    def verify_artifact(self, artifact_path: Path, expected_duration_s: float | None = None) -> dict[str, Any]:
        info = probe_artifact_ffprobe(artifact_path, self.ffprobe_bin)
        v_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not v_streams:
            raise RendererError(f"Rendered artifact has no video stream: {artifact_path}")
        if expected_duration_s is not None:
            fmt_dur = float((info.get("format") or {}).get("duration") or v_streams[0].get("duration") or 0.0)
            if fmt_dur <= 0.0:
                raise RendererError(f"Rendered artifact has zero duration: {artifact_path}")
        return info

    def _build_decode_filter_args(
        self,
        media: MediaInfo,
        plan: EditPlan,
        allow_hdr_tonemap: bool,
    ) -> list[str]:
        """Build exact FFmpeg input filtergraph for layout & color normalization without double-rotation."""
        w, h = plan.width, plan.height
        fps_str = f"{plan.fps_num}/{plan.fps_den}"
        layout = plan.layout_plan or {}
        mode = layout.get("mode", "preserve_source")

        pre_parts: list[str] = []
        if media.color_metadata.is_hdr and allow_hdr_tonemap:
            pre_parts.append(TONEMAP_VF)
        if media.is_interlaced:
            pre_parts.append("yadif=0:-1:0")

        if mode == "fit_pad_safe":
            bg_fill = layout.get("background_fill", "blur_same_media")
            pre = (",".join(pre_parts) + ",") if pre_parts else ""
            if bg_fill == "solid":
                fc = (
                    f"[0:v]{pre}fps={fps_str},"
                    f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=#101218,setsar=1,format=rgb24[outv]"
                )
            else:
                fc = (
                    f"[0:v]{pre}fps={fps_str},split=2[bg_src][fg_src];"
                    f"[bg_src]scale={w}:{h}:force_original_aspect_ratio=increase:flags=bilinear,"
                    f"crop={w}:{h},gblur=sigma=24,eq=brightness=-0.12:saturation=0.85[bg];"
                    f"[fg_src]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=rgb24[outv]"
                )
            return ["-filter_complex", fc, "-map", "[outv]"]

        vf_parts = list(pre_parts)
        vf_parts.append(f"fps={fps_str}")

        if mode == "smart_crop_tracked_subject":
            ax = float(layout.get("crop_anchor_x", 0.5))
            ay = float(layout.get("crop_anchor_y", 0.5))
            src_w, src_h = media.display_width, media.display_height
            target_ar = w / h
            src_ar = src_w / max(1, src_h)
            if src_ar > target_ar:
                crop_h = src_h
                crop_w = max(2, (int(round(src_h * target_ar)) // 2) * 2)
                cx = int(round(ax * src_w - crop_w * 0.5))
                cx = max(0, min(src_w - crop_w, cx))
                cy = 0
            else:
                crop_w = src_w
                crop_h = max(2, (int(round(src_w / target_ar)) // 2) * 2)
                cx = 0
                cy = int(round(ay * src_h - crop_h * 0.5))
                cy = max(0, min(src_h - crop_h, cy))
            vf_parts.append(f"crop={crop_w}:{crop_h}:{cx}:{cy}")
            vf_parts.append(f"scale={w}:{h}:flags=lanczos")
        else:
            if media.display_width != w or media.display_height != h or media.sar_num != media.sar_den:
                vf_parts.append(f"scale={w}:{h}:flags=lanczos")

        vf_parts.append("setsar=1")
        vf_parts.append("format=rgb24")
        return ["-vf", ",".join(vf_parts)]

    def render_visual_track(
        self,
        source_path: Path,
        media: MediaInfo,
        analysis: AnalysisResult,
        plan: EditPlan,
        ffmpeg_events: list[EffectEvent],
        out_video_only: Path,
        config: EditConfig,
        budget: ResourceBudget | None = None,
    ) -> dict[str, Any]:
        """Stream frames through deterministic effect evaluator -> video-only H.264 MP4."""
        if not self.ffmpeg_bin:
            raise RendererError("ffmpeg binary unavailable")

        t0 = time.monotonic()
        out_video_only.parent.mkdir(parents=True, exist_ok=True)
        w, h = plan.width, plan.height
        fps_str = f"{plan.fps_num}/{plan.fps_den}"
        fps_float = plan.fps_num / plan.fps_den
        frame_bytes = w * h * 3
        ff_threads = budget.snapshot.ffmpeg_threads if (budget and budget.snapshot) else 2

        # Fast-path control / passthrough when NO visual effects apply on FFmpeg AND source layout unchanged
        needs_frame_processing = bool(ffmpeg_events)

        # Build frame lookup maps for audio envelope and subject tracking
        audio_map = {pt.frame: pt.rms_norm for pt in analysis.audio_envelope}
        subj_map = {sr.frame: (sr.center_x, sr.center_y) for sr in analysis.subject_regions}
        ctx = FrameRenderContext(
            width=w,
            height=h,
            fps=fps_float,
            seed=plan.seed,
            audio_envelope_by_frame=audio_map,
            subject_track_by_frame=subj_map,
        )

        # Index events by layer priority for fast per-frame evaluation
        sorted_events = sorted(
            ffmpeg_events,
            key=lambda e: (LAYER_ORDER.get(e.layer, 99), e.priority, e.id),
        )
        needs_rolling_buffer = any(
            ev.effect_id in ("offset_layer_echo",) for ev in sorted_events
        )
        transition_starts = {
            ev.start_frame: [] for ev in sorted_events if ev.layer == "transition"
        }
        for ev in sorted_events:
            if ev.layer == "transition":
                transition_starts[ev.start_frame].append(ev.id)
        transition_ends = {ev.end_frame_exclusive: [] for ev in sorted_events if ev.layer == "transition"}
        for ev in sorted_events:
            if ev.layer == "transition":
                transition_ends[ev.end_frame_exclusive].append(ev.id)

        decode_filter_args = self._build_decode_filter_args(media, plan, config.allow_hdr_tonemap)
        dec_cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(max(1, ff_threads)),
            "-i",
            str(source_path),
            "-an",
            "-sn",
            "-dn",
            *decode_filter_args,
            "-f",
            "rawvideo",
            "pipe:1",
        ]

        preset = "fast" if config.quality == "preview" else "medium"
        crf = "22" if config.quality == "preview" else "18"
        enc_cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            fps_str,
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-threads",
            str(max(1, ff_threads)),
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-movflags",
            "+faststart",
            str(out_video_only),
        ]

        dec_proc = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        frames_written = 0
        try:
            assert dec_proc.stdout is not None
            assert enc_proc.stdin is not None
            while True:
                raw = dec_proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                if needs_frame_processing:
                    frame = frame.copy()

                    # Capture pre-cut reference frame for transitions
                    if frames_written in transition_starts:
                        for tid in transition_starts[frames_written]:
                            ctx.cut_anchor_frames[tid] = frame.copy()
                    if frames_written in transition_ends:
                        for tid in transition_ends[frames_written]:
                            ctx.cut_anchor_frames.pop(tid, None)

                    if needs_rolling_buffer:
                        ctx.recent_frames[frames_written] = frame.copy()
                        # Bound rolling buffer strictly to last 8 frames
                        stale_keys = [k for k in ctx.recent_frames if k < frames_written - 8]
                        for k in stale_keys:
                            del ctx.recent_frames[k]

                    # Apply active events in deterministic layer order
                    for ev in sorted_events:
                        if ev.start_frame <= frames_written < ev.end_frame_exclusive:
                            frame = apply_effect_frame(frame, frames_written, ev, ctx)

                enc_proc.stdin.write(frame.tobytes())
                frames_written += 1

                if budget and (frames_written % 90 == 0):
                    budget.check_runtime_disk_guard()

            enc_proc.stdin.close()
        except Exception:
            if budget:
                budget.terminate_process_tree(dec_proc)
                budget.terminate_process_tree(enc_proc)
            else:
                dec_proc.kill()
                enc_proc.kill()
            raise
        finally:
            if dec_proc.stdout:
                dec_proc.stdout.close()
            dec_err = dec_proc.stderr.read().decode("utf-8", errors="replace") if dec_proc.stderr else ""
            if dec_proc.stderr:
                dec_proc.stderr.close()
            enc_err = enc_proc.stderr.read().decode("utf-8", errors="replace") if enc_proc.stderr else ""
            if enc_proc.stderr:
                enc_proc.stderr.close()
            dec_rc = dec_proc.wait(timeout=60)
            enc_rc = enc_proc.wait(timeout=120)

        if dec_rc != 0 and frames_written == 0:
            raise RendererError(f"FFmpeg decode pipe failed (exit {dec_rc}): {dec_err.strip()}")
        if enc_rc != 0:
            raise RendererError(f"FFmpeg encode pipe failed (exit {enc_rc}): {enc_err.strip()}")
        if frames_written == 0:
            raise RendererError("Zero frames written during visual render")

        return {
            "frames_written": frames_written,
            "elapsed_s": round(time.monotonic() - t0, 4),
            "preset": preset,
            "crf": crf,
        }

    def composite_alpha_overlays(
        self,
        base_video: Path,
        overlays: list[tuple[EffectEvent, Path]],
        out_video: Path,
        plan: EditPlan,
        config: EditConfig,
        budget: ResourceBudget | None = None,
    ) -> dict[str, Any]:
        """Composite external alpha WebM overlays (Remotion / HyperFrames) onto video-only track with exact PTS shift."""
        if not overlays:
            raise ValueError("composite_alpha_overlays called with empty overlay list")
        if not self.ffmpeg_bin:
            raise RendererError("ffmpeg binary unavailable")

        t0 = time.monotonic()
        fps_float = plan.fps_num / plan.fps_den
        ff_threads = budget.snapshot.ffmpeg_threads if (budget and budget.snapshot) else 2
        preset = "fast" if config.quality == "preview" else "medium"
        crf = "22" if config.quality == "preview" else "18"

        inputs: list[str] = ["-i", str(base_video)]
        for _ev, ov_path in overlays:
            # Force libvpx-vp9 decoder so WebM yuva420p alpha channel is ALWAYS preserved!
            inputs.extend(["-c:v", "libvpx-vp9", "-i", str(ov_path)])

        filter_parts: list[str] = ["[0:v]format=yuva420p[base0]"]
        cur = "[base0]"
        for idx, (ev, _ov_path) in enumerate(overlays, start=1):
            start_s = ev.start_frame / fps_float
            end_s = ev.end_frame_exclusive / fps_float
            filter_parts.append(
                f"[{idx}:v]format=yuva420p,scale={plan.width}:{plan.height},setpts=PTS-STARTPTS+{start_s:.6f}/TB[ov{idx}]"
            )
            nxt = f"[v{idx}]"
            filter_parts.append(
                f"{cur}[ov{idx}]overlay=0:0:eof_action=pass:enable='between(t,{start_s:.6f},{end_s:.6f})'{nxt}"
            )
            cur = nxt
        filter_parts.append(f"{cur}format=yuv420p[outv]")

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(max(1, ff_threads)),
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_video),
        ]
        run_subprocess_guarded(cmd, budget=budget)
        return {"overlay_count": len(overlays), "elapsed_s": round(time.monotonic() - t0, 4)}

    def mux_with_source_audio(
        self,
        video_only_path: Path,
        source_path: Path,
        media: MediaInfo,
        plan: EditPlan,
        out_partial_mp4: Path,
        budget: ResourceBudget | None = None,
    ) -> dict[str, Any]:
        """Phase E6 HARD LOCK: Mux rendered video with original source audio master untouched.

        - Stream-copies original audio tracks whenever MP4-compatible (zero filters, zero fades, zero loudnorm).
        - Discloses transcode_compat only when source audio codec cannot legally live in MP4.
        - Never uses -shortest truncation.
        - Silent source remains silent (zero synthesized audio).
        """
        if not self.ffmpeg_bin:
            raise RendererError("ffmpeg binary unavailable")

        t0 = time.monotonic()
        out_partial_mp4.parent.mkdir(parents=True, exist_ok=True)
        policy = plan.audio_policy
        mode = str(policy.get("mode", "none"))
        selected_tracks: list[int] = list(policy.get("selected_tracks", []))

        if not media.has_audio or mode == "none" or not selected_tracks:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_only_path),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                str(out_partial_mp4),
            ]
            run_subprocess_guarded(cmd, budget=budget)
            return {"audio_mode": "none", "elapsed_s": round(time.monotonic() - t0, 4)}

        map_args: list[str] = ["-map", "0:v:0"]
        for t_idx in selected_tracks:
            map_args.extend(["-map", f"1:a:{t_idx}"])

        # Align source audio strictly by RELATIVE (audio_start - video_start) offset
        # so streams that share a non-zero container start_time (e.g. MKV/TS) stay perfectly locked.
        input_offset_args: list[str] = []
        primary_track = selected_tracks[0]
        if primary_track < len(media.audio_streams):
            rel_offset = media.audio_streams[primary_track].start_time_s - media.stream_start_time_s
            if abs(rel_offset) > 0.005:
                input_offset_args = ["-itsoffset", f"{rel_offset:.6f}"]

        if mode == "stream_copy":
            audio_codec_args = ["-c:a", "copy"]
        else:
            # Disclosed compatibility transcode preserving sample rate & channel layout, no filters/loudnorm
            audio_codec_args = ["-c:a", "aac", "-b:a", "256k", "-af", "asetpts=PTS-STARTPTS"]

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_only_path),
            *input_offset_args,
            "-i",
            str(source_path),
            *map_args,
            "-c:v",
            "copy",
            *audio_codec_args,
            "-movflags",
            "+faststart",
            str(out_partial_mp4),
        ]
        run_subprocess_guarded(cmd, budget=budget)
        return {"audio_mode": mode, "selected_tracks": selected_tracks, "elapsed_s": round(time.monotonic() - t0, 4)}

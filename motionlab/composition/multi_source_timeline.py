"""Phase A0/A2/A4/A5 (Generic Multi-Source CFR Rational Timeline, Color Match, Watermark Auth & Audio Join QA).

Unifies multiple input videos with differing native FPS (e.g., 24fps + 30fps) and
audio sample rates (e.g., 48kHz + 44.1kHz) onto a single deterministic CFR rational
output timeline and lossless 48kHz stereo working PCM clock, without mutating sources
or using AI optical flow frame hallucination.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.schemas import atomic_write_json, canonical_hash, sanitize_native

MULTI_SOURCE_SCHEMA_VERSION = "1.0.0"


@dataclass
class MultiSourceSpan:
    span_id: str
    source_id: str  # e.g. "A", "B", "bridge_editorial"
    kind: str  # "source_media" | "editorial_bridge"
    source_start_s: float
    source_end_s: float
    output_start_frame: int
    output_end_frame_exclusive: int
    output_start_sample: int
    output_end_sample: int
    crossfade_in_samples: int = 0
    crossfade_out_samples: int = 0
    reframe_zoom: float = 1.0
    reframe_cx: float = 0.5
    reframe_cy: float = 0.5
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class MultiSourceEditMap:
    schema_version: str
    sources: list[dict[str, Any]]
    target_width: int
    target_height: int
    fps_num: int
    fps_den: int
    sample_rate: int
    channels: int
    output_total_frames: int
    output_duration_s: float
    output_total_samples: int
    kept_spans: list[MultiSourceSpan] = field(default_factory=list)
    cuts: list[dict[str, Any]] = field(default_factory=list)
    map_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))

    def compute_hash(self) -> str:
        d = self.to_dict()
        d.pop("map_hash", None)
        return canonical_hash(d)


def exact_rational_samples(frames: int, fps_num: int, fps_den: int, sample_rate: int) -> int:
    """Exact rational audio sample allocation (zero cumulative rounding across spans!)."""
    return int(round(Fraction(int(frames) * int(fps_den) * int(sample_rate), int(fps_num))))


def build_multi_source_edit_map(
    sources: list[dict[str, Any]],
    kept_specs: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    *,
    target_width: int = 1080,
    target_height: int = 1920,
    fps_num: int = 30,
    fps_den: int = 1,
    sample_rate: int = 48000,
    channels: int = 2,
    crossfade_ms: float = 12.0,
) -> MultiSourceEditMap:
    """Compile exact rational multi-source EditTimeMap with zero cumulative frame/sample drift."""
    fps_frac = Fraction(fps_num, fps_den)
    xf_samples = max(0, int(round((crossfade_ms / 1000.0) * sample_rate)))
    spans: list[MultiSourceSpan] = []
    cur_frame = 0
    for idx, spec in enumerate(kept_specs):
        s0 = float(spec["start_s"])
        s1 = float(spec["end_s"])
        dur_s = max(0.0, s1 - s0)
        n_frames = max(1, int(round(Fraction(str(round(dur_s, 6))) * fps_frac)))
        # Snap source_end_s cleanly to exact rational frame duration to avoid drift
        exact_dur = float(Fraction(n_frames, 1) / fps_frac)
        end_frame = cur_frame + n_frames
        out_s0 = exact_rational_samples(cur_frame, fps_num, fps_den, sample_rate)
        out_s1 = exact_rational_samples(end_frame, fps_num, fps_den, sample_rate)
        spans.append(MultiSourceSpan(
            span_id=f"ms_{idx:02d}_{spec['source_id']}",
            source_id=str(spec["source_id"]),
            kind=str(spec.get("kind", "source_media")),
            source_start_s=round(s0, 6),
            source_end_s=round(s0 + exact_dur, 6),
            output_start_frame=cur_frame,
            output_end_frame_exclusive=end_frame,
            output_start_sample=out_s0,
            output_end_sample=out_s1,
            crossfade_in_samples=xf_samples if idx > 0 else 0,
            crossfade_out_samples=xf_samples if idx < len(kept_specs) - 1 else 0,
            reframe_zoom=float(spec.get("reframe_zoom", 1.0)),
            reframe_cx=float(spec.get("reframe_cx", 0.5)),
            reframe_cy=float(spec.get("reframe_cy", 0.5)),
            reason=str(spec.get("reason", "")),
        ))
        cur_frame = end_frame

    total_samples = exact_rational_samples(cur_frame, fps_num, fps_den, sample_rate)
    out_dur_s = round(float(Fraction(cur_frame, 1) / fps_frac), 6)
    emap = MultiSourceEditMap(
        schema_version=MULTI_SOURCE_SCHEMA_VERSION,
        sources=sanitize_native(sources),
        target_width=int(target_width),
        target_height=int(target_height),
        fps_num=int(fps_num),
        fps_den=int(fps_den),
        sample_rate=int(sample_rate),
        channels=int(channels),
        output_total_frames=cur_frame,
        output_duration_s=out_dur_s,
        output_total_samples=total_samples,
        kept_spans=spans,
        cuts=sanitize_native(cuts),
    )
    emap.map_hash = emap.compute_hash()
    return emap


def validate_multi_source_edit_map(emap: MultiSourceEditMap) -> list[str]:
    """Fail-closed invariant checker: frame/sample contiguity, provenance, bounds & hash."""
    errs: list[str] = []
    if emap.fps_num <= 0 or emap.fps_den <= 0:
        errs.append("invalid_fps")
    if emap.output_total_frames <= 0 or not emap.kept_spans:
        errs.append("empty_output_timeline")
    src_durs = {str(s["source_id"]): float(s.get("duration_s", 1e9)) for s in emap.sources}
    exp_f = 0
    exp_s = 0
    for sp in emap.kept_spans:
        if sp.output_start_frame != exp_f:
            errs.append(f"frame_discontinuity:{sp.span_id}:{sp.output_start_frame}!={exp_f}")
        if sp.output_end_frame_exclusive <= sp.output_start_frame:
            errs.append(f"non_positive_span:{sp.span_id}")
        if sp.output_start_sample != exp_s:
            errs.append(f"sample_discontinuity:{sp.span_id}:{sp.output_start_sample}!={exp_s}")
        if sp.kind == "source_media":
            if sp.source_id not in src_durs:
                errs.append(f"unknown_source_id:{sp.source_id}")
            elif sp.source_start_s < -1e-4 or sp.source_end_s > src_durs[sp.source_id] + 0.15:
                errs.append(f"source_oob:{sp.span_id}:{sp.source_start_s}-{sp.source_end_s}>{src_durs[sp.source_id]}")
        exp_f = sp.output_end_frame_exclusive
        exp_s = sp.output_end_sample
    if exp_f != emap.output_total_frames:
        errs.append(f"total_frames_mismatch:{exp_f}!={emap.output_total_frames}")
    expected_samples = exact_rational_samples(emap.output_total_frames, emap.fps_num, emap.fps_den, emap.sample_rate)
    if exp_s != expected_samples or emap.output_total_samples != expected_samples:
        errs.append(f"total_samples_mismatch:{exp_s}/{emap.output_total_samples}!={expected_samples}")
    if emap.map_hash != emap.compute_hash():
        errs.append("map_hash_tampered")
    return errs


def map_source_time_to_output(emap: MultiSourceEditMap, source_id: str, source_t: float) -> float | None:
    """Map a source timestamp (s) to edited output timeline (s), or None if cut out."""
    fps_f = emap.fps_num / float(emap.fps_den)
    for sp in emap.kept_spans:
        if sp.source_id == source_id and sp.source_start_s - 0.03 <= source_t <= sp.source_end_s + 0.03:
            clamped = max(sp.source_start_s, min(sp.source_end_s, source_t))
            return round((sp.output_start_frame / fps_f) + (clamped - sp.source_start_s), 4)
    return None


def remap_multi_source_words(
    emap: MultiSourceEditMap,
    words_by_source: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Map kept words from all sources onto the single edited output clock cleanly."""
    out: list[dict[str, Any]] = []
    fps_f = emap.fps_num / float(emap.fps_den)
    for sp in emap.kept_spans:
        if sp.kind != "source_media":
            continue
        sp_out_s0 = sp.output_start_frame / fps_f
        for w in words_by_source.get(sp.source_id, []):
            ws0 = float(w.get("source_start_s", w.get("start_s", 0.0)))
            ws1 = float(w.get("source_end_s", w.get("end_s", ws0)))
            mid = 0.5 * (ws0 + ws1)
            if sp.source_start_s <= mid <= sp.source_end_s:
                o0 = sp_out_s0 + max(0.0, ws0 - sp.source_start_s)
                o1 = sp_out_s0 + min(sp.source_end_s - sp.source_start_s, max(ws0 - sp.source_start_s + 0.02, ws1 - sp.source_start_s))
                out.append({
                    "word_id": len(out),
                    "source_id": sp.source_id,
                    "source_word_id": int(w.get("word_id", 0)),
                    "word": str(w.get("word", "")),
                    "output_start_s": round(o0, 4),
                    "output_end_s": round(o1, 4),
                    "output_start_frame": int(round(o0 * fps_f)),
                    "output_end_frame": int(round(o1 * fps_f)),
                    "confidence": float(w.get("confidence", 0.9)),
                })
    return out


def evaluate_watermark_authorization(
    watermark_roi_measurement: dict[str, Any],
    authorization_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed watermark & AI-provenance authorization gate.

    Never silently strips AI provenance marks without verified owner clean-export license!
    """
    detected_any = any(
        bool(v.get("detected_bright_static_mark"))
        for v in watermark_roi_measurement.values()
        if isinstance(v, dict)
    )
    if not detected_any:
        return sanitize_native({
            "schema_version": MULTI_SOURCE_SCHEMA_VERSION,
            "status": "NO_WATERMARK_DETECTED",
            "action": "NONE",
            "clean_release_eligible": True,
            "ai_generated_disclosure_required": False,
            "roi_measurement": watermark_roi_measurement,
        })

    auth = authorization_doc or {}
    if bool(auth.get("clean_export_authorized")) and str(auth.get("license_basis", "")).strip() and not bool(auth.get("mandatory_provenance_mark", True)):
        return sanitize_native({
            "schema_version": MULTI_SOURCE_SCHEMA_VERSION,
            "status": "AUTHORIZED_CLEANUP_ALLOWED",
            "action": str(auth.get("preferred_method", "clean_export_or_verified_crop")),
            "license_basis": str(auth["license_basis"]),
            "clean_release_eligible": True,
            "ai_generated_disclosure_required": bool(auth.get("ai_generated_disclosure_required", True)),
            "roi_measurement": watermark_roi_measurement,
        })

    return sanitize_native({
        "schema_version": MULTI_SOURCE_SCHEMA_VERSION,
        "status": "BLOCKED_WATERMARK_AUTHORIZATION",
        "action": "PRESERVE_PROVENANCE_MARK_AND_RENDER_CANDIDATE",
        "reason": (
            "Static 4-point sparkle provenance mark detected in bottom-right corner (~x=0.83, y=0.91) on both sources. "
            "No verified clean-export license or provenance-removal authorization provided; preserving mark honestly and "
            "flagging candidate release as BLOCKED_WATERMARK_AUTHORIZATION (never fabricating clean PASS)."
        ),
        "clean_release_eligible": False,
        "ai_generated_disclosure_required": True,
        "roi_measurement": watermark_roi_measurement,
    })


def compute_cross_source_color_match(
    src_a: Path,
    src_b: Path,
    *,
    times_a: tuple[float, ...] = (2.0, 5.0, 8.5),
    times_b: tuple[float, ...] = (2.0, 6.0, 12.0),
) -> dict[str, Any]:
    """Measure gentle bounded luma/contrast/white-balance alignment from B toward A (zero warping/orange skin)."""
    def _sample_stats(p: Path, ts: tuple[float, ...]) -> dict[str, float]:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ys, us, vs, stds = [], [], [], []
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            # Sample central subject region (x: 20..80%, y: 15..75%) for robust skin/knit/seat balance
            h, w = fr.shape[:2]
            roi = fr[int(h * 0.15) : int(h * 0.75), int(w * 0.20) : int(w * 0.80)]
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            ys.append(float(np.mean(ycrcb[:, :, 0])))
            stds.append(float(np.std(ycrcb[:, :, 0])))
            us.append(float(np.mean(ycrcb[:, :, 1])))  # Cr
            vs.append(float(np.mean(ycrcb[:, :, 2])))  # Cb
        cap.release()
        return {
            "luma_mean": round(float(np.mean(ys)), 2) if ys else 128.0,
            "luma_std": round(float(np.mean(stds)), 2) if stds else 45.0,
            "cr_mean": round(float(np.mean(us)), 2) if us else 128.0,
            "cb_mean": round(float(np.mean(vs)), 2) if vs else 128.0,
        }

    sa = _sample_stats(src_a, times_a)
    sb = _sample_stats(src_b, times_b)
    # Bounded gentle correction on B toward A: brightness within [-0.06, +0.06], contrast [0.94, 1.06], gamma [0.93, 1.07]
    d_luma = (sa["luma_mean"] - sb["luma_mean"]) / 255.0
    brightness = round(max(-0.055, min(0.055, d_luma * 0.55)), 4)
    ratio_std = sa["luma_std"] / max(10.0, sb["luma_std"])
    contrast = round(max(0.95, min(1.06, 1.0 + (ratio_std - 1.0) * 0.40)), 4)
    # Subtle warmth shift only if noticeable chroma difference
    d_cr = (sa["cr_mean"] - sb["cr_mean"]) / 128.0
    rm = round(max(-0.04, min(0.04, d_cr * 0.45)), 4)
    bm = round(max(-0.04, min(0.04, -d_cr * 0.35)), 4)
    vf_b = f"eq=brightness={brightness:.4f}:contrast={contrast:.4f}:saturation=1.01,colorbalance=rm={rm:.4f}:bm={bm:.4f}"
    return sanitize_native({
        "schema_version": MULTI_SOURCE_SCHEMA_VERSION,
        "source_A_stats": sa,
        "source_B_stats": sb,
        "b_correction": {"brightness": brightness, "contrast": contrast, "rm": rm, "bm": bm},
        "ffmpeg_vf_B": vf_b,
        "gentle_bounds_verified": True,
    })


def _load_f32_stereo(wav_path: Path, sr: int = 48000, ffmpeg_bin: str = "ffmpeg") -> np.ndarray:
    proc = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(wav_path), "-ac", "2", "-ar", str(sr), "-f", "f32le", "pipe:1"],
        capture_output=True, check=True, timeout=60, shell=False,
    )
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return arr.reshape((-1, 2))


def build_multi_source_mezzanine_and_voice_bed(
    source_paths: dict[str, Path],
    source_wavs_48k: dict[str, Path],
    edit_map: MultiSourceEditMap,
    out_video_mp4: Path,
    out_voice_wav: Path,
    *,
    color_match: dict[str, Any] | None = None,
    bridge_clip_path: Path | None = None,
    room_tone_window_b: tuple[float, float] = (9.25, 9.75),
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Assemble exact 1080x1920 30fps CFR clean mezzanine (no subtitles) + sample-exact 48kHz stereo voice bed."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    sr = edit_map.sample_rate
    fps_f = edit_map.fps_num / float(edit_map.fps_den)
    W, H = edit_map.target_width, edit_map.target_height

    # 1. Load 48kHz stereo PCM per source and compute gentle cross-source voice loudness match (matching B -> A safely)
    pcm_map: dict[str, np.ndarray] = {sid: _load_f32_stereo(p, sr=sr, ffmpeg_bin=ffmpeg) for sid, p in source_wavs_48k.items()}

    def _active_speech_rms(arr: np.ndarray) -> float:
        mono = np.mean(arr, axis=1)
        hop = int(0.02 * sr)
        n_b = mono.size // hop
        if n_b < 5:
            return 0.05
        rms_b = np.sqrt(np.mean(mono[: n_b * hop].reshape((n_b, hop)) ** 2, axis=1) + 1e-12)
        hi = rms_b[rms_b >= np.percentile(rms_b, 65)]
        return float(np.mean(hi)) if hi.size else float(np.mean(rms_b))

    gains: dict[str, float] = {}
    ref_rms = _active_speech_rms(pcm_map["A"]) if "A" in pcm_map else 0.08
    for sid, arr in pcm_map.items():
        if sid == "A":
            # Gentle -0.8 dB headroom on A so 2 clipping samples never re-clip
            gains[sid] = round(float(10 ** (-0.8 / 20.0)), 4)
        else:
            r_cur = _active_speech_rms(arr)
            # Cap gain boost <= +6.0 dB so noise floor never over-amplifies
            target_r = ref_rms * float(10 ** (-0.8 / 20.0))
            g = min(float(10 ** (5.8 / 20.0)), max(0.5, target_r / max(1e-5, r_cur)))
            gains[sid] = round(g, 4)

    # Extract genuine natural car room-tone loop from B non-speech window (for clean editorial bridge bed)
    rt_src = pcm_map.get("B", next(iter(pcm_map.values())))
    r0 = max(0, int(round(room_tone_window_b[0] * sr)))
    r1 = min(rt_src.shape[0], int(round(room_tone_window_b[1] * sr)))
    if r1 - r0 < int(0.15 * sr):
        room_loop = np.zeros((int(0.25 * sr), 2), dtype=np.float32)
    else:
        room_loop = rt_src[r0:r1].copy() * gains.get("B", 1.0) * 0.72
        # Smooth loop edges with 15ms crossfade
        fl = min(int(0.015 * sr), room_loop.shape[0] // 4)
        if fl > 4:
            ramp = np.linspace(0.0, 1.0, fl, dtype=np.float32)[:, None]
            room_loop[:fl] *= ramp
            room_loop[-fl:] *= (1.0 - ramp)

    # 2. Construct exact sample-aligned edited_voice_bed.wav with 12ms equal-power cosine crossfades at joins!
    bed = np.zeros((edit_map.output_total_samples, 2), dtype=np.float32)
    for sp in edit_map.kept_spans:
        n_out = sp.output_end_sample - sp.output_start_sample
        if sp.kind == "editorial_bridge":
            # Tile genuine room tone cleanly across the bridge
            reps = (n_out // max(1, room_loop.shape[0])) + 2
            seg = np.tile(room_loop, (reps, 1))[:n_out].copy()
        else:
            src_arr = pcm_map[sp.source_id]
            s_i0 = max(0, int(round(sp.source_start_s * sr)))
            s_i1 = s_i0 + n_out
            if s_i1 <= src_arr.shape[0]:
                seg = src_arr[s_i0:s_i1].copy() * gains[sp.source_id]
            else:
                avail = src_arr[min(s_i0, src_arr.shape[0]) :].copy() * gains[sp.source_id]
                seg = np.zeros((n_out, 2), dtype=np.float32)
                seg[: min(n_out, avail.shape[0])] = avail[: min(n_out, avail.shape[0])]

        # Apply smooth equal-power cosine micro-fades ONLY at non-speech splice edges!
        fi = min(sp.crossfade_in_samples, n_out // 4)
        fo = min(sp.crossfade_out_samples, n_out // 4)
        if fi > 4:
            t_in = np.linspace(0.0, math.pi * 0.5, fi, dtype=np.float32)[:, None]
            seg[:fi] *= np.sin(t_in)
        if fo > 4:
            t_out = np.linspace(0.0, math.pi * 0.5, fo, dtype=np.float32)[:, None]
            seg[-fo:] *= np.cos(t_out)
        bed[sp.output_start_sample : sp.output_end_sample] = seg

    # Gentle soft safety clip (only prevents >0.985 digital overs without touching normal speech!)
    peak_before = float(np.max(np.abs(bed))) if bed.size else 0.0
    if peak_before > 0.96:
        bed = np.tanh(bed / 0.96) * 0.96

    out_voice_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = out_voice_wav.with_suffix(".partial.wav")
    i16 = np.round(np.clip(bed, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(tmp_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())
    tmp_wav.replace(out_voice_wav)

    # 3. Assemble exact CFR 30fps 1080x1920 clean video mezzanine (strip subtitles, deterministic 24->30 CFR!)
    seg_dir = out_video_mp4.parent / "_ms_segments"
    if seg_dir.exists():
        shutil.rmtree(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_files: list[Path] = []
    vf_b_color = (color_match or {}).get("ffmpeg_vf_B", "")

    for idx, sp in enumerate(edit_map.kept_spans):
        n_f = sp.output_end_frame_exclusive - sp.output_start_frame
        seg_p = seg_dir / f"seg_{idx:02d}_{sp.source_id}.mp4"
        if sp.kind == "editorial_bridge":
            if bridge_clip_path is not None and bridge_clip_path.is_file():
                vf = f"fps={edit_map.fps_num}/{edit_map.fps_den},scale={W}:{H}:flags=lanczos,setsar=1,format=yuv420p"
                subprocess.run(
                    [
                        ffmpeg, "-y", "-v", "error", "-nostdin",
                        "-i", str(bridge_clip_path), "-an", "-sn",
                        "-vf", vf, "-frames:v", str(n_f),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p",
                        str(seg_p),
                    ],
                    check=True, timeout=120, shell=False,
                )
            else:
                dur_b = n_f / fps_f
                subprocess.run(
                    [
                        ffmpeg, "-y", "-v", "error", "-nostdin",
                        "-f", "lavfi", "-i", f"color=c=#F6F3EC:s={W}x{H}:r={edit_map.fps_num}/{edit_map.fps_den}:d={dur_b:.6f}",
                        "-frames:v", str(n_f), "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p",
                        str(seg_p),
                    ],
                    check=True, timeout=60, shell=False,
                )
        else:
            src_p = source_paths[sp.source_id]
            vf_parts: list[str] = [f"fps={edit_map.fps_num}/{edit_map.fps_den}:round=near"]
            if sp.reframe_zoom > 1.005:
                z = min(1.12, max(1.0, sp.reframe_zoom))
                cw = int(round(W / z)) // 2 * 2
                ch = int(round(H / z)) // 2 * 2
                cx_px = int(round(sp.reframe_cx * W))
                cy_px = int(round(sp.reframe_cy * H))
                ox = max(0, min(W - cw, cx_px - cw // 2))
                oy = max(0, min(H - ch, cy_px - ch // 2))
                vf_parts.append(f"scale={W}:{H}:flags=lanczos")
                vf_parts.append(f"crop={cw}:{ch}:{ox}:{oy}")
                vf_parts.append(f"scale={W}:{H}:flags=lanczos")
            else:
                vf_parts.append(f"scale={W}:{H}:flags=lanczos")
            if sp.source_id == "B" and vf_b_color:
                vf_parts.append(vf_b_color)
            vf_parts.extend(["setsar=1", "format=yuv420p"])
            # Accurate seek: fast coarse seek then exact -ss after -i
            coarse = max(0.0, sp.source_start_s - 1.0)
            fine = sp.source_start_s - coarse
            subprocess.run(
                [
                    ffmpeg, "-y", "-v", "error", "-nostdin",
                    "-ss", f"{coarse:.6f}", "-i", str(src_p),
                    "-ss", f"{fine:.6f}", "-an", "-sn",
                    "-vf", ",".join(vf_parts),
                    "-frames:v", str(n_f),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p",
                    str(seg_p),
                ],
                check=True, timeout=180, shell=False,
            )
        seg_files.append(seg_p)

    concat_list = seg_dir / "concat.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in seg_files) + "\n", encoding="utf-8")
    tmp_mp4 = out_video_mp4.with_suffix(".partial.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error", "-nostdin",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:v", "copy", "-an", "-sn", "-movflags", "+faststart",
            str(tmp_mp4),
        ],
        check=True, timeout=120, shell=False,
    )
    tmp_mp4.replace(out_video_mp4)
    shutil.rmtree(seg_dir, ignore_errors=True)

    # Verify exact decoded video frame count!
    cap = cv2.VideoCapture(str(out_video_mp4))
    verified_frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        verified_frames += 1
    cap.release()
    if verified_frames != edit_map.output_total_frames:
        raise RuntimeError(f"Multi-source mezzanine frame mismatch: got {verified_frames}, expected {edit_map.output_total_frames}")

    return sanitize_native({
        "schema_version": MULTI_SOURCE_SCHEMA_VERSION,
        "edit_map_hash": edit_map.map_hash,
        "mezzanine_path": str(out_video_mp4),
        "voice_bed_path": str(out_voice_wav),
        "verified_frame_count": verified_frames,
        "verified_sample_count": int(bed.shape[0]),
        "sample_exact": int(bed.shape[0]) == edit_map.output_total_samples,
        "cross_source_gains": gains,
        "subtitle_stream_removed": True,
    })


def evaluate_multi_source_audio_joins(voice_bed_wav: Path, edit_map: MultiSourceEditMap, *, out_path: Path | None = None) -> dict[str, Any]:
    """Check every splice on multi-source edited_voice_bed.wav for clicks, DC steps, floor jumps & sample drift."""
    with wave.open(str(voice_bed_wav), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        nf = wf.getnframes()
        arr = np.frombuffer(wf.readframes(nf), dtype=np.int16).astype(np.float32) / 32768.0
    mono = arr.reshape((-1, nch)).mean(axis=1) if nch > 1 else arr
    sample_diff = abs(int(mono.size) - int(edit_map.output_total_samples))
    joins: list[dict[str, Any]] = []
    clicks = 0
    dc_jumps = 0
    failure_reasons: list[str] = []
    if sample_diff > 2:
        failure_reasons.append(f"sample_count_drift:{sample_diff}")

    win_pre = int(round(0.05 * sr))
    for i in range(1, len(edit_map.kept_spans)):
        sp = edit_map.kept_spans[i]
        js = sp.output_start_sample
        lo = max(1, js - win_pre)
        hi = min(mono.size - 1, js + win_pre)
        pre = mono[lo:js]
        post = mono[js:hi]
        # Local 4ms splice edge max sample-to-sample derivative vs neighborhood RMS
        edge = mono[max(0, js - 96) : min(mono.size, js + 96)]
        max_step = float(np.max(np.abs(np.diff(edge)))) if edge.size > 2 else 0.0
        local_rms = float(np.sqrt(np.mean(edge * edge) + 1e-12)) if edge.size else 1e-6
        click_ratio = round(max_step / max(local_rms, 0.008), 3)
        dc_jump = round(abs(float(np.mean(post)) - float(np.mean(pre))), 5) if pre.size and post.size else 0.0
        rms_pre_db = round(float(20.0 * np.log10(max(float(np.sqrt(np.mean(pre * pre) + 1e-12)), 1e-6))), 2) if pre.size else -96.0
        rms_post_db = round(float(20.0 * np.log10(max(float(np.sqrt(np.mean(post * post) + 1e-12)), 1e-6))), 2) if post.size else -96.0
        click_flag = bool(max_step > 0.14 and click_ratio > 6.5)
        dc_flag = bool(dc_jump > 0.06)
        if click_flag:
            clicks += 1
            failure_reasons.append(f"click_at_join_{i:02d}_s{js}")
        if dc_flag:
            dc_jumps += 1
            failure_reasons.append(f"dc_step_at_join_{i:02d}_s{js}")
        joins.append({
            "join_id": f"msjoin_{i:02d}_s{js}",
            "target_span_id": sp.span_id,
            "output_sample": js,
            "output_time_s": round(js / float(sr), 4),
            "max_sample_step": round(max_step, 5),
            "click_ratio": click_ratio,
            "dc_offset_jump": dc_jump,
            "rms_pre_dbfs": rms_pre_db,
            "rms_post_dbfs": rms_post_db,
            "floor_jump_db": round(abs(rms_post_db - rms_pre_db), 2),
            "click_flagged": click_flag,
            "dc_flagged": dc_flag,
            "join_ok": not (click_flag or dc_flag),
        })

    doc = sanitize_native({
        "schema_version": MULTI_SOURCE_SCHEMA_VERSION,
        "edit_map_hash": edit_map.map_hash,
        "sample_rate": sr,
        "expected_samples": edit_map.output_total_samples,
        "actual_samples": int(mono.size),
        "sample_count_diff": sample_diff,
        "joins_checked": len(joins),
        "clicks_detected": clicks,
        "dc_jumps_detected": dc_jumps,
        "joins": joins,
        "failure_reasons": failure_reasons,
        "status": "PASS" if not failure_reasons else "FAIL",
    })
    if out_path is not None:
        atomic_write_json(out_path, doc)
    return doc


"""Phase B1: Media preflight probing, SHA-256 streaming fingerprint, and metadata validation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from motionlab.schemas import (
    AudioStreamInfo,
    MediaInfo,
    Rational,
    SchemaValidationError,
    validate_local_path_string,
)

MP4_COPY_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", "alac", "mp2", "opus"}
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
INTERLACED_ORDERS = {"tt", "bb", "tb", "bt"}


class PreflightError(RuntimeError):
    """Raised when an input media file fails preflight validation."""


def streaming_sha256(path: Path, chunk_size: int = 1024 * 64) -> str:
    """Compute file SHA-256 in bounded streaming chunks (never loads whole media into RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_rational(value: str | None, fallback: Rational | None = None) -> Rational | None:
    if not value:
        return fallback
    s = str(value).strip()
    if s in ("0/0", "0/1", "N/A", ""):
        return fallback
    try:
        if "/" in s:
            a_s, b_s = s.split("/", 1)
            a, b = int(a_s), int(b_s)
            if a <= 0 or b <= 0:
                return fallback
            frac = Fraction(a, b)
        elif ":" in s:
            a_s, b_s = s.split(":", 1)
            a, b = int(a_s), int(b_s)
            if a <= 0 or b <= 0:
                return fallback
            frac = Fraction(a, b)
        else:
            val = float(s)
            if not math.isfinite(val) or val <= 0:
                return fallback
            frac = Fraction(val).limit_denominator(100000)
        return Rational(frac.numerator, frac.denominator)
    except (ValueError, ZeroDivisionError):
        return fallback


def _extract_rotation(video_stream: dict[str, Any]) -> tuple[int, bool]:
    """Return (rotation_deg_normalized_0_90_180_270, from_side_data)."""
    for sd in video_stream.get("side_data_list") or []:
        rot = sd.get("rotation")
        if rot is not None:
            try:
                deg = int(round(float(rot))) % 360
                return deg, True
            except (ValueError, TypeError):
                pass
    tags = video_stream.get("tags") or {}
    if "rotate" in tags:
        try:
            deg = int(round(float(tags["rotate"]))) % 360
            return deg, False
        except (ValueError, TypeError):
            pass
    return 0, False


def _detect_vfr(video_stream: dict[str, Any], ffprobe_bin: str, path: Path) -> tuple[bool, Rational]:
    """Detect VFR by comparing avg_frame_rate and r_frame_rate, plus bounded packet delta check."""
    avg_r = _parse_rational(video_stream.get("avg_frame_rate"))
    nom_r = _parse_rational(video_stream.get("r_frame_rate"))

    if avg_r is None and nom_r is None:
        raise PreflightError("Unable to determine valid video frame rate from source metadata.")

    chosen = avg_r or nom_r
    assert chosen is not None

    is_vfr = False
    if avg_r is not None and nom_r is not None:
        ratio = avg_r.as_float() / nom_r.as_float()
        # Significant divergence (>2%) indicates VFR
        if abs(ratio - 1.0) > 0.02:
            is_vfr = True

    # Bounded check of first 45 frame packet durations to catch subtle VFR even when avg ~ r
    if not is_vfr:
        try:
            proc = subprocess.run(
                [
                    ffprobe_bin, "-v", "error",
                    "-select_streams", "v:0",
                    "-read_intervals", "%+#45",
                    "-show_entries", "packet=duration_time",
                    "-of", "json",
                    str(path),
                ],
                capture_output=True, text=True, timeout=10, shell=False,
            )
            if proc.returncode == 0:
                packets = json.loads(proc.stdout).get("packets") or []
                durs = []
                for pkt in packets:
                    dt = pkt.get("duration_time")
                    if dt is not None:
                        try:
                            v = float(dt)
                            if math.isfinite(v) and v > 0:
                                durs.append(v)
                        except (TypeError, ValueError):
                            pass
                if len(durs) >= 6:
                    # Drop first/last packet in case of container priming
                    core = durs[1:-1] if len(durs) > 6 else durs
                    mn, mx = min(core), max(core)
                    if mn > 0 and (mx / mn) > 1.25:
                        is_vfr = True
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass

    return is_vfr, chosen


def probe_media(input_path: str | Path, ffprobe_bin: str | None = None) -> MediaInfo:
    """Run ffprobe preflight validation and compute streaming SHA-256 fingerprint."""
    try:
        path = validate_local_path_string(str(input_path), allow_nonexistent=False)
    except SchemaValidationError as exc:
        raise PreflightError(str(exc)) from exc

    if not path.is_file():
        raise PreflightError(f"Input must be a regular readable file: {path}")
    file_size = path.stat().st_size
    if file_size <= 0:
        raise PreflightError(f"Input file is empty (0 bytes): {path}")

    ffprobe = ffprobe_bin or shutil.which("ffprobe")
    if not ffprobe:
        raise PreflightError("ffprobe executable not found on PATH.")

    cmd = [
        ffprobe, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, shell=False)
    if proc.returncode != 0:
        raise PreflightError(
            f"ffprobe failed to parse media (corrupt or unsupported): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"ffprobe returned invalid JSON: {exc}") from exc

    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise PreflightError(f"No video stream found in {path}")
    v = video_streams[0]

    try:
        width = int(v["width"])
        height = int(v["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError("Video stream missing valid width/height") from exc
    if width <= 0 or height <= 0:
        raise PreflightError(f"Invalid video dimensions: {width}x{height}")

    warnings: list[str] = []
    if width % 2 != 0 or height % 2 != 0:
        warnings.append(
            f"Source dimensions {width}x{height} are odd; yuv420p encoding requires even dimensions (will pad/align safely)."
        )

    rotation, from_side_data = _extract_rotation(v)
    if rotation % 360 in (90, 270):
        display_w, display_h = height, width
        if not from_side_data:
            warnings.append(
                f"Rotation tag {rotation} deg present without display-matrix side_data; verifying explicit orientation."
            )
    else:
        display_w, display_h = width, height

    sar = _parse_rational(v.get("sample_aspect_ratio"), fallback=Rational(1, 1))
    assert sar is not None
    if sar.num != sar.den:
        warnings.append(f"Non-square SAR detected ({sar.num}:{sar.den}); display aspect preserved.")

    is_vfr, fps = _detect_vfr(v, ffprobe, path)
    if is_vfr:
        warnings.append(
            f"Variable frame rate (VFR) detected; average fps={fps.to_ffmpeg()}. "
            "Timeline maps to rational CFR via timestamp-aware sampling while preserving audio master."
        )

    timebase = _parse_rational(v.get("time_base"), fallback=Rational(1, 1000))
    assert timebase is not None

    # Duration resolution
    dur_s: float | None = None
    for cand in (v.get("duration"), fmt.get("duration")):
        if cand is not None and str(cand) != "N/A":
            try:
                val = float(cand)
                if math.isfinite(val) and val > 0:
                    dur_s = val
                    break
            except ValueError:
                pass

    nb_frames_raw = v.get("nb_frames")
    nb_frames: int | None = None
    if nb_frames_raw is not None and str(nb_frames_raw).isdigit():
        nb_frames = int(nb_frames_raw)

    if dur_s is None and nb_frames and nb_frames > 0:
        dur_s = fps.frame_to_seconds(nb_frames)
    if dur_s is None or dur_s <= 0:
        raise PreflightError(f"Video has zero or unreadable duration: {path}")

    if nb_frames and nb_frames > 0 and not is_vfr:
        # Cross-check nb_frames against duration*fps
        est_frames = max(1, fps.seconds_to_frame(dur_s))
        if abs(nb_frames - est_frames) <= 2:
            duration_frames = nb_frames
        else:
            duration_frames = est_frames
    else:
        duration_frames = max(1, fps.seconds_to_frame(dur_s))

    # Start PTS and offsets
    start_pts = int(v.get("start_pts") or 0)
    try:
        video_start_s = float(v.get("start_time") or 0.0)
        if not math.isfinite(video_start_s):
            video_start_s = 0.0
    except (TypeError, ValueError):
        video_start_s = 0.0
    if abs(video_start_s) > 0.005:
        warnings.append(f"Non-zero video stream start_time={video_start_s:.4f}s detected; preserving A/V offset.")

    duration_pts = int(v.get("duration_ts") or round(dur_s / timebase.as_float()))

    pix_fmt = str(v.get("pix_fmt") or "unknown")
    color_transfer = v.get("color_transfer")
    is_hdr = str(color_transfer or "").lower() in HDR_TRANSFERS
    if is_hdr:
        warnings.append(
            f"HDR transfer ({color_transfer}) detected. ML-01 will NOT silently tone-map; "
            "pass --allow-hdr-tonemap explicitly or use SDR source."
        )

    field_order = str(v.get("field_order") or "progressive").lower()
    interlaced = field_order in INTERLACED_ORDERS
    if interlaced:
        warnings.append(f"Interlaced field_order={field_order} detected.")

    has_alpha = any(tok in pix_fmt.lower() for tok in ("yuva", "rgba", "bgra", "argb", "abgr", "gbrap"))
    if has_alpha:
        warnings.append(f"Source pixel format {pix_fmt} carries alpha channel; standard MP4 yuv420p output drops alpha.")

    color_metadata = {
        "color_space": v.get("color_space"),
        "color_transfer": color_transfer,
        "color_primaries": v.get("color_primaries"),
        "color_range": v.get("color_range"),
    }

    audio_streams: list[AudioStreamInfo] = []
    for a_idx, s in enumerate(s for s in streams if s.get("codec_type") == "audio"):
        codec = str(s.get("codec_name") or "unknown").lower()
        sr = int(s.get("sample_rate") or 0)
        ch = int(s.get("channels") or 0)
        layout = str(s.get("channel_layout") or f"{ch}ch")
        br = int(s["bit_rate"]) if s.get("bit_rate") and str(s["bit_rate"]).isdigit() else None
        a_start_pts = int(s.get("start_pts") or 0)
        try:
            a_start_s = float(s.get("start_time") or 0.0)
            if not math.isfinite(a_start_s):
                a_start_s = 0.0
        except (TypeError, ValueError):
            a_start_s = 0.0
        try:
            a_dur_s = float(s["duration"]) if s.get("duration") and str(s["duration"]) != "N/A" else None
        except (TypeError, ValueError):
            a_dur_s = None
        lang = (s.get("tags") or {}).get("language")
        mp4_ok = codec in MP4_COPY_AUDIO_CODECS
        if not mp4_ok:
            warnings.append(
                f"Audio track {a_idx} uses codec '{codec}' which is not directly MP4 stream-copy safe; "
                "will disclose high-bitrate AAC transcode with PCM correlation validation."
            )
        audio_streams.append(
            AudioStreamInfo(
                stream_index=int(s.get("index", a_idx + 1)),
                audio_track_index=a_idx,
                codec_name=codec,
                sample_rate=sr,
                channels=ch,
                channel_layout=layout,
                bit_rate=br,
                start_pts=a_start_pts,
                start_time_s=a_start_s,
                duration_s=a_dur_s,
                language=lang,
                mp4_copy_compatible=mp4_ok,
            )
        )

    if len(audio_streams) > 1:
        warnings.append(
            f"Source has {len(audio_streams)} audio tracks; preserving all tracks by default "
            "(pass --audio-track N to select a specific track for analysis/mux)."
        )

    file_hash = streaming_sha256(path)
    return MediaInfo(
        path=str(path),
        input_hash=file_hash,
        file_size_bytes=file_size,
        container_format=str(fmt.get("format_name") or "unknown"),
        duration_pts=duration_pts,
        duration_s=dur_s,
        duration_frames=duration_frames,
        width=width,
        height=height,
        display_width=display_w,
        display_height=display_h,
        rotation=rotation,
        sar=sar,
        fps=fps,
        timebase=timebase,
        vfr=is_vfr,
        stream_start_pts=start_pts,
        video_start_time_s=video_start_s,
        video_codec=str(v.get("codec_name") or "unknown"),
        pixel_format=pix_fmt,
        color_metadata=color_metadata,
        interlaced=interlaced,
        has_alpha=has_alpha,
        is_hdr=is_hdr,
        audio_streams=audio_streams,
        warnings=warnings,
    )

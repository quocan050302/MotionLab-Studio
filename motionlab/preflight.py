"""Phase B1 — Preflight media probing, binary discovery, streaming SHA256, and strict input checks."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from motionlab.schemas import AudioStreamInfo, ColorMetadata, MediaInfo

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
ALPHA_PIX_FMTS = {"yuva420p", "yuva422p", "yuva444p", "rgba", "bgra", "argb", "abgr", "gbrap", "ya8"}
MP4_AUDIO_COPY_CODECS = {"aac", "mp3", "ac3", "eac3", "alac", "opus"}
SUPPORTED_VIDEO_CODECS = {
    "h264",
    "hevc",
    "h265",
    "vp8",
    "vp9",
    "av1",
    "mpeg4",
    "prores",
    "mjpeg",
    "png",
    "ffv1",
    "rawvideo",
    "mpeg2video",
    "theora",
}


class PreflightError(RuntimeError):
    """Actionable failure during media preflight."""


def find_binary(name: str) -> str | None:
    return shutil.which(name)


def sha256_streaming(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash without reading the entire media file into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def parse_rational(raw: str | None, default: tuple[int, int] = (30, 1)) -> Fraction:
    if not raw or raw in ("0/0", "N/A", "0"):
        return Fraction(default[0], default[1])
    try:
        if ":" in raw:
            a, b = raw.split(":", 1)
            f = Fraction(int(a), int(b))
        elif "/" in raw:
            a, b = raw.split("/", 1)
            if int(b) == 0:
                return Fraction(default[0], default[1])
            f = Fraction(int(a), int(b))
        else:
            f = Fraction(raw).limit_denominator(100000)
        if f <= 0:
            return Fraction(default[0], default[1])
        return f
    except (ValueError, ZeroDivisionError):
        return Fraction(default[0], default[1])


def _extract_rotation(stream: dict[str, Any]) -> int:
    # Prefer display-matrix side_data rotation (FFmpeg autorotates this)
    for sd in stream.get("side_data_list") or []:
        if sd.get("rotation") is not None:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (ValueError, TypeError):
                pass
    # Fallback tag inspection (warn-worthy if present without side data)
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(round(float(tags["rotate"]))) % 360
        except (ValueError, TypeError):
            pass
    return 0


def probe_media(
    path: Path,
    ffprobe_bin: str | None = None,
    allow_hdr_tonemap: bool = False,
) -> MediaInfo:
    """Run ffprobe JSON and validate regular file, streams, codecs, rotation, SAR, VFR, HDR, audio."""
    resolved = path.resolve()
    if not resolved.exists():
        raise PreflightError(f"Input file does not exist: {resolved}")
    if not resolved.is_file():
        raise PreflightError(f"Input path is not a regular file: {resolved}")
    file_size = resolved.stat().st_size
    if file_size <= 0:
        raise PreflightError(f"Input file is empty (0 bytes): {resolved}")

    ffprobe = ffprobe_bin or find_binary("ffprobe")
    if not ffprobe:
        raise PreflightError("ffprobe not found on PATH. Install FFmpeg and run `python -m motionlab doctor`.")

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(resolved),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise PreflightError(
            f"ffprobe failed on '{resolved.name}' (exit {proc.returncode}): {proc.stderr.strip() or 'corrupt or unreadable media'}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"ffprobe returned invalid JSON for '{resolved.name}'") from exc

    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video_streams = [s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")]
    if not video_streams:
        raise PreflightError(f"No playable video stream found in '{resolved.name}'")

    v = video_streams[0]
    codec_name = str(v.get("codec_name") or "unknown").lower()
    if codec_name not in SUPPORTED_VIDEO_CODECS:
        raise PreflightError(f"Unsupported video codec '{codec_name}' in '{resolved.name}'")

    width = int(v.get("width") or 0)
    height = int(v.get("height") or 0)
    if width <= 0 or height <= 0:
        raise PreflightError(f"Invalid video dimensions {width}x{height} in '{resolved.name}'")

    warnings: list[str] = []
    if width % 2 != 0 or height % 2 != 0:
        warnings.append(f"odd_source_dimensions:{width}x{height}")

    rotation = _extract_rotation(v)
    sar = parse_rational(v.get("sample_aspect_ratio"), default=(1, 1))
    display_w, display_h = width, height
    if rotation in (90, 270):
        display_w, display_h = display_h, display_w
    if sar != Fraction(1, 1):
        display_w = int(round(display_w * float(sar)))
        warnings.append(f"non_square_sar:{sar.numerator}:{sar.denominator}")

    # Parse rational FPS and detect VFR
    avg_raw = v.get("avg_frame_rate")
    r_raw = v.get("r_frame_rate")
    has_avg = bool(avg_raw and avg_raw not in ("0/0", "0"))
    has_r = bool(r_raw and r_raw not in ("0/0", "0"))

    if not has_avg and not has_r:
        warnings.append("unknown_source_fps_fallback_30")
        fps = Fraction(30, 1)
        vfr = False
    else:
        fps_avg = parse_rational(avg_raw, default=(30, 1)) if has_avg else None
        fps_r = parse_rational(r_raw, default=(30, 1)) if has_r else None
        fps = fps_avg or fps_r or Fraction(30, 1)
        if fps > Fraction(240, 1):
            # Extremely high r_frame_rate on VFR container — prefer avg if reasonable
            fps = fps_avg if (fps_avg and fps_avg <= Fraction(240, 1)) else Fraction(30, 1)
            vfr = True
        elif fps_avg and fps_r and abs(float(fps_avg) - float(fps_r)) / max(float(fps_r), 1e-6) > 0.02:
            vfr = True
        else:
            vfr = False
    if vfr:
        warnings.append(f"vfr_detected:avg={avg_raw},r={r_raw}")

    tb = parse_rational(v.get("time_base"), default=(1, 90000))
    # Duration: prefer stream duration; when falling back to container format.duration
    # (e.g. Matroska/MKV with non-zero start_time), subtract container start_time so
    # playable duration reflects actual content length rather than end-PTS.
    stream_dur_raw = v.get("duration")
    fmt_dur_raw = fmt.get("duration")
    try:
        fmt_start_s = float(fmt.get("start_time") or 0.0)
    except (ValueError, TypeError):
        fmt_start_s = 0.0
    try:
        if stream_dur_raw is not None:
            duration_s = float(stream_dur_raw)
        elif fmt_dur_raw is not None:
            raw_fd = float(fmt_dur_raw)
            duration_s = (raw_fd - fmt_start_s) if (fmt_start_s > 0.005 and raw_fd > fmt_start_s) else raw_fd
        else:
            duration_s = 0.0
    except (ValueError, TypeError):
        duration_s = 0.0

    nb_frames_raw = v.get("nb_frames")
    nb_frames = 0
    if nb_frames_raw is not None:
        try:
            nb_frames = int(nb_frames_raw)
        except (ValueError, TypeError):
            nb_frames = 0

    if duration_s <= 0.0 and nb_frames > 0:
        duration_s = float(nb_frames / fps)
    if duration_s <= 0.0 or not math.isfinite(duration_s):
        raise PreflightError(f"Zero or invalid video duration ({dur_raw!r}) in '{resolved.name}'")

    if nb_frames <= 0:
        nb_frames = max(1, int(round(duration_s * float(fps))))

    dur_ts_raw = v.get("duration_ts")
    if dur_ts_raw is not None:
        try:
            duration_pts = int(dur_ts_raw)
        except (ValueError, TypeError):
            duration_pts = int(round(duration_s / float(tb)))
    else:
        duration_pts = int(round(duration_s / float(tb)))

    start_pts = int(v.get("start_pts") or 0)
    try:
        start_time_s = float(v.get("start_time") or 0.0)
    except (ValueError, TypeError):
        start_time_s = 0.0
    if abs(start_time_s) > 0.005 or start_pts != 0:
        warnings.append(f"nonzero_video_start_pts:{start_pts} ({start_time_s:.4f}s)")

    pix_fmt = str(v.get("pix_fmt") or "unknown")
    has_alpha = pix_fmt.lower() in ALPHA_PIX_FMTS
    if has_alpha:
        warnings.append(f"source_has_alpha:{pix_fmt}")

    field_order = str(v.get("field_order") or "progressive")
    is_interlaced = field_order.lower() in ("tt", "bb", "tb", "bt")
    if is_interlaced:
        warnings.append(f"interlaced_source:{field_order}")

    color_transfer = str(v.get("color_transfer") or "unknown")
    color_space = str(v.get("color_space") or "unknown")
    color_primaries = str(v.get("color_primaries") or "unknown")
    color_range = str(v.get("color_range") or "unknown")
    is_hdr = color_transfer.lower() in HDR_TRANSFERS
    if is_hdr:
        if not allow_hdr_tonemap:
            warnings.append(f"hdr_detected_without_tonemap_opt_in:{color_transfer}")
        else:
            warnings.append(f"hdr_explicit_tonemap_enabled:{color_transfer}")

    color_meta = ColorMetadata(
        color_space=color_space,
        color_transfer=color_transfer,
        color_primaries=color_primaries,
        color_range=color_range,
        is_hdr=is_hdr,
    )

    # Probe all audio streams (do NOT assume stream 0 is voice)
    audio_streams: list[AudioStreamInfo] = []
    audio_idx = 0
    for s in streams:
        if s.get("codec_type") != "audio":
            continue
        acodec = str(s.get("codec_name") or "unknown").lower()
        sr = int(s.get("sample_rate") or 0)
        ch = int(s.get("channels") or 0)
        ch_layout = str(s.get("channel_layout") or f"{ch}ch")
        try:
            adur = float(s.get("duration") or duration_s)
        except (ValueError, TypeError):
            adur = duration_s
        astart_pts = int(s.get("start_pts") or 0)
        try:
            astart_t = float(s.get("start_time") or 0.0)
        except (ValueError, TypeError):
            astart_t = 0.0
        br = None
        if s.get("bit_rate"):
            try:
                br = int(s["bit_rate"])
            except (ValueError, TypeError):
                br = None
        init_pad = int(s.get("initial_padding") or 0)
        lang = str((s.get("tags") or {}).get("language") or "und")
        copy_ok = acodec in MP4_AUDIO_COPY_CODECS
        if not copy_ok:
            warnings.append(f"audio_track_{audio_idx}_requires_mp4_transcode:{acodec}")
        if abs(astart_t - start_time_s) > 0.02:
            warnings.append(f"audio_track_{audio_idx}_start_offset:{astart_t:.4f}s")
        audio_streams.append(
            AudioStreamInfo(
                index=int(s.get("index", audio_idx + 1)),
                audio_index=audio_idx,
                codec_name=acodec,
                sample_rate=sr,
                channels=ch,
                channel_layout=ch_layout,
                duration_s=adur,
                start_pts=astart_pts,
                start_time_s=astart_t,
                time_base=str(s.get("time_base") or f"1/{max(sr, 1)}"),
                bit_rate=br,
                initial_padding=init_pad,
                language=lang,
                mp4_copy_compatible=copy_ok,
            )
        )
        audio_idx += 1

    if not audio_streams:
        warnings.append("no_audio_stream_valid_silent_input")

    input_hash = sha256_streaming(resolved)

    return MediaInfo(
        source_path=str(resolved),
        input_hash=input_hash,
        file_size_bytes=file_size,
        duration_pts=duration_pts,
        duration_s=duration_s,
        time_base_num=tb.numerator,
        time_base_den=tb.denominator,
        width=width,
        height=height,
        display_width=display_w,
        display_height=display_h,
        rotation=rotation,
        sar_num=sar.numerator,
        sar_den=sar.denominator,
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        vfr=vfr,
        nb_frames=nb_frames,
        stream_start_pts=start_pts,
        stream_start_time_s=start_time_s,
        video_codec=codec_name,
        pixel_format=pix_fmt,
        field_order=field_order,
        is_interlaced=is_interlaced,
        has_alpha=has_alpha,
        color_metadata=color_meta,
        audio_streams=audio_streams,
        warnings=warnings,
    )

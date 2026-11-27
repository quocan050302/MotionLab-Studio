"""Legal, deterministic procedural video/audio fixture generators (zero external media)."""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from motionlab.preflight import find_binary


def _ffmpeg() -> str:
    exe = find_binary("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg required to generate procedural test fixtures")
    return exe


def _write_pulsed_speech_like_wav(wav_path: Path, duration_s: float, sr: int = 44100, freq_hz: float = 220.0) -> None:
    """Generate deterministic audio with distinct energy impulses every ~2.2s for sync & onset testing."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    # Base low bed + harmonic bursts spaced every 2.2s so onset/peak detector & PCM sync lag are unambiguous
    env = np.full(n, 0.04, dtype=np.float32)
    for beat_t in np.arange(1.0, max(1.1, duration_s - 0.6), 2.2):
        center = int(beat_t * sr)
        width = int(0.28 * sr)
        a = max(0, center - width // 2)
        b = min(n, center + width // 2)
        if b > a:
            local = np.linspace(0.0, math.pi, b - a, dtype=np.float32)
            env[a:b] = np.maximum(env[a:b], 0.04 + 0.68 * np.sin(local) ** 2)
    sig = env * (
        0.55 * np.sin(2.0 * math.pi * freq_hz * t)
        + 0.30 * np.sin(2.0 * math.pi * (freq_hz * 1.5) * t)
        + 0.15 * np.sin(2.0 * math.pi * (freq_hz * 2.5) * t)
    )
    pcm = np.clip(sig * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_multishot_demo(
    out_mp4: Path,
    duration_s: float = 15.0,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    with_audio: bool = True,
) -> Path:
    """Create a rich, deterministic multi-shot video with real hard cuts, moving shapes, and optional sync audio."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(duration_s * fps)))
    ff = _ffmpeg()

    wav_path = out_mp4.with_suffix(".tmp_audio.wav")
    if with_audio:
        _write_pulsed_speech_like_wav(wav_path, duration_s=duration_s)

    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
    ]
    if with_audio:
        cmd.extend(["-i", str(wav_path), "-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
    )

    # Precompute 3 visually distinct shot bases so hard cuts at 1/3 and 2/3 are unmistakable
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    shot_bases = []
    # Shot 0: Cool slate blue studio gradient
    s0 = np.zeros((height, width, 3), dtype=np.uint8)
    s0[:, :, 0] = np.clip(35 + 55 * xx, 0, 255).astype(np.uint8)
    s0[:, :, 1] = np.clip(58 + 75 * yy, 0, 255).astype(np.uint8)
    s0[:, :, 2] = np.clip(115 + 95 * (1.0 - yy), 0, 255).astype(np.uint8)
    shot_bases.append(s0)

    # Shot 1: Warm terracotta / amber studio gradient
    s1 = np.zeros((height, width, 3), dtype=np.uint8)
    s1[:, :, 0] = np.clip(165 + 65 * (1.0 - yy), 0, 255).astype(np.uint8)
    s1[:, :, 1] = np.clip(82 + 75 * xx, 0, 255).astype(np.uint8)
    s1[:, :, 2] = np.clip(48 + 45 * yy, 0, 255).astype(np.uint8)
    shot_bases.append(s1)

    # Shot 2: Deep emerald / teal studio gradient
    s2 = np.zeros((height, width, 3), dtype=np.uint8)
    s2[:, :, 0] = np.clip(32 + 45 * yy, 0, 255).astype(np.uint8)
    s2[:, :, 1] = np.clip(118 + 82 * (1.0 - xx), 0, 255).astype(np.uint8)
    s2[:, :, 2] = np.clip(95 + 75 * yy, 0, 255).astype(np.uint8)
    shot_bases.append(s2)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        cut1 = total_frames // 3
        cut2 = (2 * total_frames) // 3
        for f in range(total_frames):
            shot_idx = 0 if f < cut1 else (1 if f < cut2 else 2)
            frame = shot_bases[shot_idx].copy()
            t = f / max(1, fps)

            # Subtle reference grid lines & gentle moving geometric subject proxy
            cx = int(width * (0.50 + 0.03 * math.sin(t * 0.9)))
            cy = int(height * (0.44 + 0.02 * math.cos(t * 0.7)))
            rad = max(24, min(width, height) // 7)
            cv2.circle(frame, (cx, cy), rad, (236, 206, 184), -1)
            cv2.rectangle(
                frame,
                (cx - int(rad * 0.9), cy + int(rad * 1.05)),
                (cx + int(rad * 0.9), min(height - 10, cy + int(rad * 2.3))),
                (44, 54, 78) if shot_idx != 1 else (70, 48, 42),
                -1,
            )
            # Highlight orb so optical glow / light sweep are measurable
            cv2.circle(frame, (int(width * 0.24), int(height * 0.20)), max(12, rad // 3), (248, 245, 232), -1)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait(timeout=120)
        if rc != 0:
            raise RuntimeError(f"Failed generating procedural demo video: {err}")
    finally:
        if wav_path.exists():
            wav_path.unlink(missing_ok=True)
    return out_mp4


def make_landscape_screen_ui(out_mp4: Path, duration_s: float = 4.0, width: int = 960, height: int = 540, fps: int = 24) -> Path:
    """Create a crisp high-edge landscape screen-recording/UI fixture to verify `tutorial` safe protection."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    total_frames = max(2, int(round(duration_s * fps)))
    base = np.full((height, width, 3), 24, dtype=np.uint8)
    # Dense crisp UI chrome along top bar, left sidebar, bottom status bar, and text-like code lines
    cv2.rectangle(base, (0, 0), (width, 46), (40, 44, 52), -1)
    cv2.rectangle(base, (0, 46), (190, height - 30), (30, 34, 42), -1)
    cv2.rectangle(base, (0, height - 30), (width, height), (36, 98, 165), -1)
    for row in range(65, height - 45, 16):
        cv2.rectangle(base, (18, row), (165, row + 6), (170, 180, 195), -1)
        line_w = 220 + ((row * 37) % 420)
        cv2.rectangle(base, (220, row), (min(width - 25, 220 + line_w), row + 5), (210, 220, 235), -1)
    for col in range(16, width - 20, 48):
        cv2.rectangle(base, (col, 14), (col + 28, 32), (185, 195, 210), 1)

    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for f in range(total_frames):
        fr = base.copy()
        # Tiny cursor move
        cx = 260 + (f * 2) % 300
        cy = 180 + (f % 40)
        cv2.circle(fr, (cx, cy), 4, (255, 220, 80), -1)
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait(timeout=60)
    return out_mp4


def make_multitrack_rotated(out_mp4: Path, duration_s: float = 3.0) -> Path:
    """Create a 2-audio-track video with genuine display-matrix rotation side-data (90) to verify track selection & rotation."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    w0 = out_mp4.with_suffix(".t0.wav")
    w1 = out_mp4.with_suffix(".t1.wav")
    raw_mp4 = out_mp4.with_suffix(".unrot.mp4")
    _write_pulsed_speech_like_wav(w0, duration_s, freq_hz=260.0)
    _write_pulsed_speech_like_wav(w1, duration_s, freq_hz=520.0)
    try:
        cmd1 = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=24:duration={duration_s}",
            "-i", str(w0),
            "-i", str(w1),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            str(raw_mp4),
        ]
        subprocess.run(cmd1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # Stamp display-matrix rotation side_data cleanly on stream copy
        cmd2 = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-noautorotate", "-display_rotation:v:0", "90",
            "-i", str(raw_mp4),
            "-map", "0", "-c", "copy",
            str(out_mp4),
        ]
        subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    finally:
        w0.unlink(missing_ok=True)
        w1.unlink(missing_ok=True)
        raw_mp4.unlink(missing_ok=True)
    return out_mp4


def make_pcm_mkv_and_offset(out_mkv: Path, duration_s: float = 3.0, ts_offset_s: float = 0.4) -> Path:
    """Create MKV with raw PCM audio (requires disclosed MP4 transcode) + non-zero start PTS."""
    out_mkv.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    wav = out_mkv.with_suffix(".src.wav")
    _write_pulsed_speech_like_wav(wav, duration_s, freq_hz=310.0)
    try:
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=360x640:rate=24:duration={duration_s}",
            "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le",
            "-output_ts_offset", f"{ts_offset_s:.3f}",
            str(out_mkv),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    finally:
        wav.unlink(missing_ok=True)
    return out_mkv


def make_short_clip(out_mp4: Path, frames: int = 8, fps: int = 24) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    dur = frames / fps
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0x336699:s=360x640:r={fps}:d={dur:.4f}",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_mp4

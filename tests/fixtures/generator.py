"""Deterministic legal procedural video/audio fixture generator for ML-01 testing and demo verification.

All generated clips are 100% original procedural test patterns (zero third-party footage, zero copyrighted audio).
Synthetic audio is strictly for sync, envelope, and audio-integrity testing, NOT a substitute for human voice perceptual review.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np


def _write_speech_like_wav(
    wav_path: Path,
    duration_s: float,
    sr: int = 48000,
    channels: int = 2,
    impulse_times_s: tuple[float, ...] = (1.5, 4.0, 6.8, 9.2, 12.1),
    quiet_scale: float = 1.0,
) -> None:
    """Synthesize a clean modulated formant-like acoustic envelope with explicit transient sync markers."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / sr

    # Syllabic amplitude envelope (~3.2 Hz modulation + brief pauses)
    env = 0.25 + 0.55 * np.maximum(0.0, np.sin(2.0 * math.pi * 1.45 * t)) * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * 4.2 * t)
    )
    # Carve brief quiet pause gaps (e.g. 4.6-5.2s, 9.6-10.2s)
    for ps, pe in ((4.5, 5.1), (9.5, 10.1)):
        if ps < duration_s:
            env[(t >= ps) & (t <= min(pe, duration_s))] *= 0.05

    # Warm multi-harmonic carrier (190Hz + 380Hz + 760Hz)
    sig = (
        0.50 * np.sin(2.0 * math.pi * 190.0 * t)
        + 0.30 * np.sin(2.0 * math.pi * 380.0 * t)
        + 0.15 * np.sin(2.0 * math.pi * 760.0 * t)
    ) * env * 0.38 * quiet_scale

    # Add crisp known acoustic sync transients at impulse_times_s
    for imp_t in impulse_times_s:
        if imp_t + 0.12 < duration_s:
            idx0 = int(round(imp_t * sr))
            seg_len = int(round(0.10 * sr))
            tau = np.arange(seg_len, dtype=np.float32) / sr
            burst = np.exp(-tau * 35.0) * np.sin(2.0 * math.pi * 880.0 * tau) * 0.55 * quiet_scale
            sig[idx0 : idx0 + seg_len] += burst

    sig = np.clip(sig, -0.92, 0.92)
    pcm16 = np.round(sig * 32767.0).astype(np.int16)
    if channels == 2:
        stereo = np.stack([pcm16, pcm16], axis=1).reshape(-1)
        raw = stereo.tobytes()
    else:
        raw = pcm16.tobytes()

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


def _render_demo_frame(f_idx: int, total_f: int, w: int, h: int, fps: int) -> np.ndarray:
    """Render 3 distinct visual scenes across 15s (cuts at 1/3 and 2/3) with realistic depth, gradients, and motion."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cut1 = total_f // 3
    cut2 = (2 * total_f) // 3
    t_s = f_idx / max(1, fps)

    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    if f_idx < cut1:
        # Shot 1: Warm studio backdrop + centered calm subject silhouette/avatar
        img[..., 0] = np.clip(28 + 35 * (1 - ys) + 18 * xs, 0, 255).astype(np.uint8)
        img[..., 1] = np.clip(36 + 44 * (1 - ys) + 12 * (1 - xs), 0, 255).astype(np.uint8)
        img[..., 2] = np.clip(58 + 52 * (1 - ys), 0, 255).astype(np.uint8)
        # Soft studio key-light backdrop orb
        cv2.circle(img, (int(w * 0.50), int(h * 0.36)), int(min(w, h) * 0.36), (62, 74, 102), -1)
        # Subtle animated subject breathing motion
        bob = int(round(math.sin(t_s * 2.0) * h * 0.004))
        cx, cy = w // 2, int(h * 0.42) + bob
        # Shoulders & torso
        cv2.ellipse(img, (cx, int(h * 0.72) + bob), (int(w * 0.30), int(h * 0.22)), 0, 0, 360, (44, 54, 72), -1)
        # Head + neutral stylized studio avatar
        head_r = int(min(w, h) * 0.13)
        cv2.circle(img, (cx, cy), head_r, (226, 186, 158), -1)
        cv2.circle(img, (cx - int(head_r * 0.32), cy - int(head_r * 0.12)), max(3, head_r // 12), (38, 36, 44), -1)
        cv2.circle(img, (cx + int(head_r * 0.32), cy - int(head_r * 0.12)), max(3, head_r // 12), (38, 36, 44), -1)
        cv2.ellipse(img, (cx, cy + int(head_r * 0.28)), (int(head_r * 0.22), int(head_r * 0.08)), 0, 10, 170, (145, 86, 78), max(2, head_r // 18))
        # Studio desk prop
        cv2.rectangle(img, (int(w * 0.14), int(h * 0.80)), (int(w * 0.86), int(h * 0.84)), (88, 96, 118), -1)

    elif f_idx < cut2:
        # Shot 2: Clean teal/slate product showcase table with moving glossy object
        img[..., 0] = np.clip(18 + 25 * xs + 22 * ys, 0, 255).astype(np.uint8)
        img[..., 1] = np.clip(64 + 48 * (1 - ys) + 18 * xs, 0, 255).astype(np.uint8)
        img[..., 2] = np.clip(78 + 56 * (1 - xs), 0, 255).astype(np.uint8)
        # Architectural grid lines on floor
        for gy in range(int(h * 0.62), h, max(24, h // 18)):
            cv2.line(img, (0, gy), (w, gy), (52, 92, 106), 1)
        # Elevated product plinth
        px0, py0, px1, py1 = int(w * 0.22), int(h * 0.30), int(w * 0.78), int(h * 0.66)
        shift_x = int(round(math.sin((t_s - 5.0) * 1.1) * w * 0.02))
        cv2.rectangle(img, (px0 + shift_x, py0), (px1 + shift_x, py1), (235, 240, 246), -1)
        cv2.rectangle(img, (px0 + shift_x + 12, py0 + 12), (px1 + shift_x - 12, py1 - 12), (34, 42, 54), -1)
        # Geometric emblem inside showcase
        cv2.circle(img, (w // 2 + shift_x, (py0 + py1) // 2), int(min(w, h) * 0.11), (242, 152, 68), -1)
        cv2.circle(img, (w // 2 + shift_x, (py0 + py1) // 2), int(min(w, h) * 0.06), (255, 232, 198), -1)

    else:
        # Shot 3: Warm sunset amber/indigo architectural scene with drifting horizon light
        img[..., 0] = np.clip(145 + 75 * (1 - ys), 0, 255).astype(np.uint8)
        img[..., 1] = np.clip(78 + 64 * (1 - ys) + 25 * xs, 0, 255).astype(np.uint8)
        img[..., 2] = np.clip(62 + 58 * ys, 0, 255).astype(np.uint8)
        # Sun orb
        sun_x = int(w * (0.42 + 0.06 * math.sin((t_s - 10.0) * 0.8)))
        cv2.circle(img, (sun_x, int(h * 0.34)), int(min(w, h) * 0.16), (255, 224, 165), -1)
        # Layered dark horizon silhouettes
        pts1 = np.array([[0, int(h * 0.64)], [int(w * 0.35), int(h * 0.56)], [int(w * 0.72), int(h * 0.62)], [w, int(h * 0.54)], [w, h], [0, h]], np.int32)
        cv2.fillPoly(img, [pts1], (52, 38, 64))
        pts2 = np.array([[0, int(h * 0.76)], [int(w * 0.45), int(h * 0.70)], [w, int(h * 0.78)], [w, h], [0, h]], np.int32)
        cv2.fillPoly(img, [pts2], (28, 22, 40))

    return img


def generate_demo_9x16_15s(out_path: Path, width: int = 1080, height: int = 1920, fps: int = 30, duration_s: float = 15.0) -> Path:
    """Generate a fresh 15s 9:16 1080x1920 30fps multi-shot fixture with AAC audio and sync transients."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg required"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_f = int(round(duration_s * fps))

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        wav_path = td_p / "audio.wav"
        _write_speech_like_wav(wav_path, duration_s=duration_s, sr=48000, channels=2)

        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        assert proc.stdin is not None
        try:
            for f in range(total_f):
                frame = _render_demo_frame(f, total_f, width, height, fps)
                proc.stdin.write(frame.tobytes())
        finally:
            proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait(timeout=120)
        if rc != 0:
            raise RuntimeError(f"Fixture generation failed: {err}")
    return out_path


def generate_silent_fixture(out_path: Path, width: int = 540, height: int = 960, fps: int = 30, duration_s: float = 4.0) -> Path:
    """Generate a valid no-audio MP4 clip."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_f = int(round(duration_s * fps))
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-an", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    assert proc.stdin is not None
    for f in range(total_f):
        proc.stdin.write(_render_demo_frame(f, total_f, width, height, fps).tobytes())
    proc.stdin.close()
    proc.wait(timeout=60)
    return out_path


def generate_screen_tutorial_fixture(out_path: Path, width: int = 1280, height: int = 720, fps: int = 30, duration_s: float = 6.0) -> Path:
    """Generate a landscape screen-recording proxy with crisp UI chrome, top/bottom bars, and text blocks."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_f = int(round(duration_s * fps))

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "aud.wav"
        _write_speech_like_wav(wav_path, duration_s=duration_s, impulse_times_s=(1.2, 3.4, 5.0))
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        assert proc.stdin is not None
        for f in range(total_f):
            img = np.full((height, width, 3), (24, 28, 36), dtype=np.uint8)
            # Crisp top toolbar and bottom status bar with dense fine UI lines
            cv2.rectangle(img, (0, 0), (width, 72), (38, 44, 58), -1)
            cv2.rectangle(img, (0, height - 80), (width, height), (34, 40, 52), -1)
            cv2.rectangle(img, (0, 72), (210, height - 80), (30, 35, 46), -1)
            for bx in range(16, width - 24, 26):
                cv2.rectangle(img, (bx, 18), (bx + 18, 52), (210, 220, 238), 2)
                cv2.line(img, (bx, height - 58), (bx + 18, height - 58), (195, 210, 230), 2)
                cv2.line(img, (bx, height - 34), (bx + 18, height - 34), (160, 180, 210), 2)
            # Code-like high-contrast text lines
            for row in range(105, height - 110, 26):
                line_w = 240 + ((row * 37) % 520)
                cv2.line(img, (240, row), (240 + line_w, row), (215, 225, 240), 2)
            # Moving cursor
            cx = 320 + int(300 * (f / max(1, total_f)))
            cy = 240 + int(120 * math.sin(f * 0.12))
            cv2.circle(img, (cx, cy), 8, (110, 220, 255), -1)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        proc.wait(timeout=60)
    return out_path


def generate_multitrack_fixture(out_path: Path, duration_s: float = 4.0) -> Path:
    """Generate an MP4 with 2 distinct audio tracks (track 0 quiet ambience, track 1 louder voice proxy)."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base_v = td_p / "v.mp4"
        generate_silent_fixture(base_v, width=480, height=360, fps=30, duration_s=duration_s)
        w0 = td_p / "track0.wav"
        w1 = td_p / "track1.wav"
        _write_speech_like_wav(w0, duration_s=duration_s, impulse_times_s=(), quiet_scale=0.15)
        _write_speech_like_wav(w1, duration_s=duration_s, impulse_times_s=(0.8, 2.1, 3.2), quiet_scale=0.95)
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(base_v), "-i", str(w0), "-i", str(w1),
                "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(out_path),
            ],
            check=True, shell=False,
        )
    return out_path


def generate_rotated_fixture(out_path: Path, duration_s: float = 3.0) -> Path:
    """Generate a coded-landscape clip with 90-degree display rotation metadata."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        base_v = Path(td) / "base.mp4"
        generate_silent_fixture(base_v, width=640, height=360, fps=30, duration_s=duration_s)
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-display_rotation:v:0", "90",
                "-i", str(base_v),
                "-c", "copy",
                "-metadata:s:v:0", "rotate=90",
                str(out_path),
            ],
            check=True, shell=False,
        )
    return out_path


def generate_pcm_mov_fixture(out_path: Path, duration_s: float = 3.0) -> Path:
    """Generate a MOV with raw PCM audio to verify disclosed AAC transcode + PCM correlation check."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        base_v = td_p / "base.mp4"
        wav_p = td_p / "pcm.wav"
        generate_silent_fixture(base_v, width=480, height=360, fps=30, duration_s=duration_s)
        _write_speech_like_wav(wav_p, duration_s=duration_s, impulse_times_s=(0.7, 1.8, 2.5))
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(base_v), "-i", str(wav_p),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "pcm_s16le",
                str(out_path),
            ],
            check=True, shell=False,
        )
    return out_path

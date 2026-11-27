"""A7 Unit Tests: Foley Asset Provenance, Rational Clock, Voice-Priority Mix & Negative Audio QA (ML-01.R5)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from motionlab.audio.assets import inspect_foley_library
from motionlab.audio.cues import frame_to_sample_exact
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.audio.schema import AUDIO_SCHEMA_VERSION, AudioTimeline, SoundEvent
from motionlab.audio.synthesis import build_sfx_asset_manifest, synthesize_sfx_waveform
from motionlab.preflight import probe_media


def _make_synthetic_voice_mp4(path: Path, dur_s: float = 4.0, sr: int = 44100) -> Path:
    """Generate a clean synthetic speech-like MP4 with known AAC mono audio track."""
    ff = shutil.which("ffmpeg")
    assert ff, "ffmpeg required"
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    # Speech-like modulated voiced harmonics around -18 dBFS
    env = 0.14 * (0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t))
    sig = env * (0.6 * np.sin(2 * np.pi * 210 * t) + 0.3 * np.sin(2 * np.pi * 420 * t) + 0.15 * np.sin(2 * np.pi * 840 * t))
    wav_p = path.parent / "_syn_voice.wav"
    with wave.open(str(wav_p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.round(np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(
        [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x292822:s=360x640:d={dur_s}:r=24",
            "-i", str(wav_p),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(path),
        ],
        check=True,
    )
    wav_p.unlink(missing_ok=True)
    return path


def test_rational_frame_to_sample_clock_exact() -> None:
    assert frame_to_sample_exact(24, 24, 1, 44100) == 44100
    assert frame_to_sample_exact(30, 30000, 1001, 48000) == 48048
    assert frame_to_sample_exact(60, 60, 1, 48000) == 48000


def test_foley_manifest_honest_status_and_tamper_rejection(tmp_path: Path) -> None:
    # Default repo manifest honestly reports PARTIAL_FOLEY_ASSETS (zero mislabeled recorded claims!)
    rep = inspect_foley_library()
    assert rep["status"] in ("PARTIAL_FOLEY_ASSETS", "RECORDED_FOLEY_VERIFIED")
    sfx_m = build_sfx_asset_manifest(sr=44100, seed=42)
    if rep["recorded_assets_verified"] == 0:
        assert all(a["provenance"] == "procedural_dsp_synthesis" for a in sfx_m["assets"])

    # Create temporary foley folder with bad hash, bad license, and path traversal -> ALL rejected!
    fdir = tmp_path / "foley"
    fdir.mkdir()
    wav_p = fdir / "click.wav"
    with wave.open(str(wav_p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.zeros(4410, dtype=np.int16).tobytes())
    good_sha = hashlib.sha256(wav_p.read_bytes()).hexdigest()

    manifest = {
        "assets": [
            {"asset_id": "bad_sha", "relative_path": "click.wav", "family": "cash_flick", "variant": 0, "license": "CC0-1.0", "sha256": "0" * 64},
            {"asset_id": "bad_lic", "relative_path": "click.wav", "family": "cash_flick", "variant": 1, "license": "ALL_RIGHTS_RESERVED", "sha256": good_sha},
            {"asset_id": "bad_trav", "relative_path": "../escape.wav", "family": "cash_flick", "variant": 2, "license": "CC0-1.0", "sha256": good_sha},
            {"asset_id": "good_one", "relative_path": "click.wav", "family": "cash_flick", "variant": 0, "license": "CC0-1.0", "sha256": good_sha},
        ]
    }
    (fdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = inspect_foley_library(fdir)
    assert out["recorded_assets_verified"] == 1
    assert len(out["rejected_assets"]) == 3


def test_voice_priority_mix_audibility_and_silent_negative(tmp_path: Path) -> None:
    ff = shutil.which("ffmpeg")
    assert ff
    src_mp4 = _make_synthetic_voice_mp4(tmp_path / "src.mp4", dur_s=4.0)
    media = probe_media(src_mp4)
    sr = 44100
    wav, spec = synthesize_sfx_waveform("cash_flick", variant=0, sr=sr, seed=42)
    ps = int(round(1.5 * sr))
    ts = int(round(spec.transient_offset_s * sr))
    ev = SoundEvent(
        id="sfx_01", linked_visual_event_id="s1", semantic_span_ids=[], observed_action_id="s1",
        asset_id=spec.asset_id, family="cash_flick", variant=0, asset_sha256=spec.sha256,
        start_sample=ps - ts, peak_sample=ps, end_sample=ps - ts + wav.size, peak_frame=36,
        gain_db=-4.5, pan=0.0, role="cash_flick", confidence=0.95, reason="test_cue",
    )
    tl = AudioTimeline(
        schema_version=AUDIO_SCHEMA_VERSION, source_hash=media.input_hash, audio_mode="mix",
        sfx_enabled=True, music_enabled=False, sample_rate=sr, channels=2,
        duration_samples=int(round(media.duration_s * sr)), fps_num=24, fps_den=1,
        pts_priming_offset_s=0.0, events=[ev],
    )
    run_d = tmp_path / "run_ok"
    mix_info = mix_preserving_master(media, tl, None, run_d, seed=42)
    assert mix_info["master_exact"] is True
    assert mix_info["metrics"]["music_enabled"] is False
    assert mix_info["metrics"]["clipping_samples_added"] == 0

    fin_mp4 = run_d / "final.mp4"
    subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src_mp4), "-i", mix_info["mixed_reference_path"],
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(fin_mp4)],
        check=True,
    )
    _checks, qa = validate_final_audio_mix(media, fin_mp4, tl, mix_info, run_d)
    assert qa["status"] == "PASS"
    assert qa["events_proxy_audible"] == 1

    # Negative test: passing source MP4 (no SFX mixed!) MUST FAIL audibility QA!
    _neg_checks, neg_qa = validate_final_audio_mix(media, src_mp4, tl, mix_info, run_d)
    assert neg_qa["status"] == "FAIL"
    assert neg_qa["events_proxy_audible"] == 0


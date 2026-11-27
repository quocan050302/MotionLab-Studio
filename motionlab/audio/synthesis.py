"""Phase A4 (ML-01.R5): Upgraded Physical-Modeling Local Foley/SFX DSP Synthesis + Recorded Asset Loader.

Every procedural sound is explicitly marked `provenance="procedural_dsp_synthesis"` (NEVER mislabeled
as studio-recorded foley). Designed for warm editorial finance/product audio clarity on mobile speakers:
crisp tactile banknote/paper transients, sweeping page turns, woody settle ticks, velvet transitions,
and restrained warm low-end payoff impacts — zero harsh casino coins or meme effects.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from motionlab.audio.assets import inspect_foley_library, load_recorded_foley_pcm
from motionlab.audio.schema import SfxAssetSpec

SFX_FAMILIES: dict[str, str] = {
    "cash_flick": "Crisp dry tactile banknote snap + fibrous paper flutter (banknote release / allocation)",
    "paper_rustle": "Soft brushed organic paper shuffle (handling / category movement)",
    "envelope_slide": "Smooth cardstock paper glide + gentle pocket settle",
    "page_turn": "Tactile 3D workbook page turn with resonant mid-band paper sweep + spine crease",
    "soft_whoosh": "Velvet low-mid editorial air sweep aligned to transition peak",
    "reverse_whoosh": "Gentle pre-reveal swell rising cleanly into visual settle",
    "pop_reveal": "Warm wooden/marimba organic pop for card or label reveal",
    "tick_accent": "Crisp muted mechanical/wooden tick for exact numeric counter landing",
    "soft_impact": "Restrained warm felt low-end thump for main contrast or $100 payoff",
    "shimmer_payoff": "Delicate warm fifth-interval harmonic chime tail for closing/payoff",
    "riser_build": "Subtle warm breath riser leading into reveal beat",
    "bass_settle": "Clean sub-bass warm editorial resolution settle",
    "ambience_pad": "Quiet subtle warm air bed (kept low, opt-in)",
}


def _bandpass(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * sr
    lo = max(20.0, min(lo_hz, nyq * 0.88)) / nyq
    hi = max(lo + 0.02, min(hi_hz, nyq * 0.96)) / nyq
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _lowpass(x: np.ndarray, sr: int, cut_hz: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * sr
    sos = butter(order, max(40.0, min(cut_hz, nyq * 0.94)) / nyq, btype="lowpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _highpass(x: np.ndarray, sr: int, cut_hz: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    sos = butter(order, max(20.0, min(cut_hz, nyq * 0.9)) / nyq, btype="highpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _stereo_micro_width(mono: np.ndarray, sr: int, rng: np.random.Generator, width: float = 0.22) -> np.ndarray:
    delay = max(1, int(round(sr * 0.0007)))
    r = np.roll(mono, delay)
    r[:delay] = 0.0
    n_diff = rng.normal(0.0, 0.015, size=mono.shape).astype(np.float32) * (np.abs(mono) > 1e-4)
    left = mono + width * n_diff
    right = (1.0 - width * 0.3) * mono + (width * 0.3) * r - width * n_diff
    return np.stack([left, right], axis=1).astype(np.float32)


def _normalize_stereo(st: np.ndarray, target_peak: float = 0.88) -> np.ndarray:
    peak = float(np.max(np.abs(st))) if st.size else 0.0
    if peak < 1e-6:
        return st.astype(np.float32)
    return (st * (target_peak / peak)).astype(np.float32)


def synthesize_sfx_variant(
    family: str,
    variant: int = 0,
    sr: int = 44100,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Generate crisp, organic physical-modeling stereo SFX PCM (float32, N x 2) and transient_offset_s."""
    # Check verified local recorded foley library first!
    lib = inspect_foley_library()
    for a in lib.get("assets", []):
        if a.get("family") == family and int(a.get("variant", 0)) == int(variant):
            loaded = load_recorded_foley_pcm(a, target_sr=sr)
            if loaded is not None:
                return loaded

    rng = np.random.default_rng(int(seed) * 1009 + sum(ord(c) for c in family) * 31 + int(variant) * 137)
    pitch_shift = 1.0 + ((variant % 3) - 1) * 0.06

    if family == "cash_flick":
        dur = 0.24
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        raw = rng.normal(0.0, 1.0, n).astype(np.float32)
        # Double-stage crisp banknote snap: lead pinch (0.018s) + snappy fibrous flick (0.048s)
        env1 = np.exp(-((t - 0.018) / 0.008) ** 2) * 0.45
        env2 = np.where(t >= 0.042, np.exp(-(t - 0.042) / 0.032) * (1.0 - np.exp(-(t - 0.042) / 0.0025)), 0.0)
        crisp = _bandpass(raw * (env1 + env2), sr, 1800 * pitch_shift, 6800 * pitch_shift)
        body = _bandpass(raw * env2, sr, 360 * pitch_shift, 1150 * pitch_shift) * 0.85
        thump = np.sin(2 * np.pi * (240 * pitch_shift) * np.maximum(0.0, t - 0.042)) * np.exp(-np.maximum(0.0, t - 0.042) / 0.014) * (t >= 0.042) * 0.35
        mono = crisp * 1.25 + body + thump
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.24), 0.90), 0.046

    if family == "paper_rustle":
        dur = 0.34
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        raw = rng.normal(0.0, 1.0, n).astype(np.float32)
        mod = 0.55 + 0.45 * np.sin(2 * np.pi * (17.0 * pitch_shift) * t + 0.6)
        env = np.sin(np.pi * np.clip(t / dur, 0.0, 1.0)) ** 1.4 * mod
        mono = _bandpass(raw * env, sr, 950 * pitch_shift, 5200 * pitch_shift) * 1.1 + _bandpass(raw * env, sr, 320, 900) * 0.45
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.30), 0.86), 0.13

    if family in ("envelope_slide", "page_turn"):
        dur = 0.44 if family == "page_turn" else 0.36
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        u = t / dur
        raw = rng.normal(0.0, 1.0, n).astype(np.float32)
        env_lo = np.sin(np.pi * u) ** 1.3
        env_hi = np.exp(-((u - 0.42) / 0.19) ** 2)
        crease = np.exp(-((u - 0.34) / 0.035) ** 2) * 0.85
        lo_band = _bandpass(raw * env_lo, sr, 420 * pitch_shift, 1450 * pitch_shift)
        hi_band = _bandpass(raw * (env_hi + crease), sr, 1750 * pitch_shift, 6200 * pitch_shift)
        settle = _bandpass(raw * np.exp(-((u - 0.78) / 0.06) ** 2), sr, 600, 2200) * 0.55
        mono = lo_band * 0.9 + hi_band * 1.15 + settle
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.32), 0.88), round(dur * 0.40, 4)

    if family == "soft_whoosh":
        dur = 0.42
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        u = t / dur
        raw = rng.normal(0.0, 1.0, n).astype(np.float32)
        env = np.exp(-((u - 0.42) / 0.24) ** 2)
        air_lo = _bandpass(raw * env, sr, 160 * pitch_shift, 680 * pitch_shift)
        air_mid = _bandpass(raw * (env ** 1.4), sr, 650 * pitch_shift, 2100 * pitch_shift) * 0.65
        tone = np.sin(2 * np.pi * (140 + 180 * u) * t) * env * 0.14
        mono = air_lo * 1.1 + air_mid + tone
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.36), 0.86), 0.17

    if family == "reverse_whoosh":
        dur = 0.38
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        u = t / dur
        raw = rng.normal(0.0, 1.0, n).astype(np.float32)
        env = (u ** 2.1) * np.where(u < 0.90, 1.0, np.clip((1.0 - u) / 0.10, 0.0, 1.0))
        mono = _bandpass(raw * env, sr, 240 * pitch_shift, 1950 * pitch_shift) + _lowpass(raw * env, sr, 320) * 0.6
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.32), 0.86), 0.32

    if family == "tick_accent":
        dur = 0.12
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        # Crisp tactile wooden marimba/typewriter tick (clearly audible on phone speakers!)
        f1 = 1180.0 * pitch_shift
        f2 = 2480.0 * pitch_shift
        f3 = 480.0 * pitch_shift
        env1 = np.exp(-t / 0.018) * (1.0 - np.exp(-t / 0.0012))
        env2 = np.exp(-t / 0.008)
        click = _bandpass(rng.normal(0.0, 1.0, n).astype(np.float32) * np.exp(-t / 0.004), sr, 2200, 6500) * 0.45
        mono = (np.sin(2 * np.pi * f1 * t) * env1 * 0.85 + np.sin(2 * np.pi * f2 * t) * env2 * 0.45 + np.sin(2 * np.pi * f3 * t) * np.exp(-t / 0.024) * 0.55 + click)
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.16), 0.88), 0.004

    if family == "pop_reveal":
        dur = 0.18
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        freq = (260.0 + 340.0 * np.exp(-t / 0.022)) * pitch_shift
        phase = 2 * np.pi * np.cumsum(freq) / sr
        env = np.exp(-t / 0.036) * (1.0 - np.exp(-t / 0.002))
        overtone = np.sin(2 * np.pi * (880.0 * pitch_shift) * t) * np.exp(-t / 0.016) * 0.32
        mono = np.sin(phase) * env + overtone
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.18), 0.88), 0.006

    if family == "soft_impact":
        dur = 0.52
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        # Warm felt editorial low-end thump + woody attack (clean mobile speaker translation via 140Hz harmonic!)
        freq = (96.0 * np.exp(-t / 0.075) + 46.0) * pitch_shift
        phase = 2 * np.pi * np.cumsum(freq) / sr
        sub = np.sin(phase) * np.exp(-t / 0.14) * (1.0 - np.exp(-t / 0.003))
        punch = np.sin(2 * np.pi * (175.0 * pitch_shift) * t) * np.exp(-t / 0.045) * 0.65
        felt = _bandpass(rng.normal(0.0, 1.0, n).astype(np.float32) * np.exp(-t / 0.022), sr, 240, 1400) * 0.45
        mono = sub + punch + felt
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.18), 0.90), 0.008

    if family == "shimmer_payoff":
        dur = 0.62
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        env = (1.0 - np.exp(-t / 0.018)) * np.exp(-t / 0.22)
        # Warm major-fifth bell harmonics (C5 + G5 + D6 — restrained editorial warmth)
        mono = (
            0.55 * np.sin(2 * np.pi * 523.25 * pitch_shift * t)
            + 0.35 * np.sin(2 * np.pi * 783.99 * pitch_shift * t)
            + 0.18 * np.sin(2 * np.pi * 1174.66 * pitch_shift * t)
        ) * env
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.35), 0.80), 0.028

    if family == "riser_build":
        dur = 0.55
        n = int(round(sr * dur))
        t = np.arange(n, dtype=np.float32) / sr
        u = t / dur
        raw = rng.normal(0.0, 1.0, n).astype(np.float32)
        env = (u ** 1.8) * np.where(u < 0.92, 1.0, np.clip((1.0 - u) / 0.08, 0.0, 1.0))
        mono = _bandpass(raw * env, sr, 260, 2200) + np.sin(2 * np.pi * (180 + 220 * u) * t) * env * 0.25
        return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.28), 0.82), 0.46

    # bass_settle / ambience_pad default
    dur = 0.48
    n = int(round(sr * dur))
    t = np.arange(n, dtype=np.float32) / sr
    freq = (82.0 * np.exp(-t / 0.09) + 52.0) * pitch_shift
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = (1.0 - np.exp(-t / 0.006)) * np.exp(-t / 0.16)
    warm = np.sin(2 * np.pi * (156.0 * pitch_shift) * t) * np.exp(-t / 0.06) * 0.42
    mono = np.sin(phase) * env + warm
    return _normalize_stereo(_stereo_micro_width(mono, sr, rng, 0.16), 0.85), 0.012


def synthesize_sfx_waveform(
    family: str,
    variant: int = 0,
    sr: int = 44100,
    seed: int = 42,
) -> tuple[np.ndarray, SfxAssetSpec]:
    """Backward-compatible R4/R5 helper returning (mono float32 waveform, typed SfxAssetSpec)."""
    foley_lib = inspect_foley_library()
    rec = None
    for a in foley_lib.get("assets", []):
        if a.get("family") == family and int(a.get("variant", 0)) == int(variant):
            rec = a
            break
    pcm_st, t_off = synthesize_sfx_variant(family, variant=variant, sr=sr, seed=seed)
    mono = (0.5 * (pcm_st[:, 0] + pcm_st[:, 1])).astype(np.float32)
    peak = float(np.max(np.abs(pcm_st))) if pcm_st.size else 0.0
    peak_db = round(20.0 * float(np.log10(max(1e-7, peak))), 2)
    sha = hashlib.sha256(pcm_st.tobytes()).hexdigest()
    spec = SfxAssetSpec(
        asset_id=f"sfx_{family}_v{variant}",
        family=family,
        variant=int(variant),
        provenance="recorded_local_foley" if rec else "procedural_dsp_synthesis",
        license=str(rec["license"]) if rec else "MotionLab Local Physical-Modeling DSP (MIT)",
        attribution=str(rec.get("attribution", "")) if rec else "Synthesized deterministically offline via scipy/numpy physical modeling",
        sample_rate=sr,
        channels=2,
        duration_s=round(pcm_st.shape[0] / sr, 4),
        transient_offset_s=round(t_off, 4),
        peak_dbfs=peak_db,
        sha256=sha,
        description=SFX_FAMILIES.get(family, family),
    )
    return mono, spec


def build_sfx_asset_manifest(sr: int = 44100, seed: int = 42) -> dict[str, Any]:
    """Generate complete deterministic SFX/foley provenance manifest honestly separating recorded vs DSP."""
    foley_lib = inspect_foley_library()
    recorded_map = {(a["family"], int(a.get("variant", 0))): a for a in foley_lib.get("assets", [])}
    assets: list[dict[str, Any]] = []
    for fam, desc in SFX_FAMILIES.items():
        for var in (0, 1, 2):
            pcm, t_off = synthesize_sfx_variant(fam, variant=var, sr=sr, seed=seed)
            sha = hashlib.sha256(pcm.tobytes()).hexdigest()
            peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
            peak_db = round(20.0 * float(np.log10(max(1e-7, peak))), 2)
            rec = recorded_map.get((fam, var))
            spec = SfxAssetSpec(
                asset_id=f"sfx_{fam}_v{var}",
                family=fam,
                variant=var,
                provenance="recorded_local_foley" if rec else "procedural_dsp_synthesis",
                license=str(rec["license"]) if rec else "MotionLab Local Physical-Modeling DSP (MIT)",
                attribution=str(rec.get("attribution", "")) if rec else "Synthesized deterministically offline via scipy/numpy physical modeling",
                sample_rate=sr,
                channels=2,
                duration_s=round(pcm.shape[0] / sr, 4),
                transient_offset_s=round(t_off, 4),
                peak_dbfs=peak_db,
                sha256=sha,
                description=desc,
            )
            assets.append(spec.to_dict())
    return {
        "schema_version": "2.0.0",
        "foley_library_status": foley_lib["status"],
        "recorded_assets_verified": foley_lib["recorded_assets_verified"],
        "total_assets": len(assets),
        "families": sorted(SFX_FAMILIES.keys()),
        "notes": foley_lib["notes"],
        "assets": assets,
    }

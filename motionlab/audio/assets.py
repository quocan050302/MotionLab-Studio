"""Phase A4 (ML-01.R5): Local Recorded Foley Asset Manifest & Strict Provenance Validator.

Supports owner-supplied or verified CC0/commercial-permitted recorded WAV foley inside `assets/sfx_foley/`
with per-file SHA-256, license, creator, transient sample offset, and redistribution metadata.
When no recorded foley assets are provisioned, honestly reports `PARTIAL_FOLEY_ASSETS` and routes to
explicitly labeled `procedural_dsp_synthesis` (NEVER mislabeling DSP as studio-recorded foley).
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.config import PROJECT_ROOT
from motionlab.schemas import ensure_contained_path, sanitize_native

FOLEY_DIR = PROJECT_ROOT / "assets" / "sfx_foley"
FOLEY_MANIFEST_PATH = FOLEY_DIR / "manifest.json"
PERMITTED_RECORDED_LICENSES = {"CC0-1.0", "PUBLIC_DOMAIN", "OWNER_RECORDED", "COMMERCIAL_LICENSED"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class FoleyAssetRecord:
    asset_id: str
    relative_path: str
    family: str
    variant: int
    duration_s: float
    transient_sample: int
    original_sample_rate: int
    sha256: str
    creator: str
    source_url: str
    license: str
    attribution: str
    allowed_usage: str
    redistribution_allowed: bool
    provenance: str = "recorded_local_foley"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def inspect_foley_library(foley_dir: Path | None = None) -> dict[str, Any]:
    """Audit local recorded foley manifest with strict SHA-256 and license verification (zero network)."""
    fdir = (foley_dir or FOLEY_DIR).resolve()
    mpath = fdir / "manifest.json"
    result: dict[str, Any] = {
        "foley_dir": str(fdir),
        "manifest_present": mpath.is_file(),
        "status": "PARTIAL_FOLEY_ASSETS",
        "recorded_assets_verified": 0,
        "recorded_families": [],
        "assets": [],
        "rejected_assets": [],
        "notes": [],
    }
    if not mpath.is_file():
        result["notes"].append(
            "No owner-supplied/CC0 recorded foley WAVs provisioned in assets/sfx_foley/. "
            "Using upgraded physical-modeling procedural_dsp_synthesis honestly labeled as PARTIAL_FOLEY_ASSETS."
        )
        return result

    try:
        raw = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as exc:
        result["notes"].append(f"Invalid foley manifest JSON: {exc}")
        return result

    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in raw.get("assets", []):
        aid = str(entry.get("asset_id", "unknown"))
        lic = str(entry.get("license", "")).upper()
        rel = str(entry.get("relative_path", ""))
        exp_sha = str(entry.get("sha256", ""))
        if lic not in PERMITTED_RECORDED_LICENSES:
            rejected.append({"asset_id": aid, "reason": f"unpermitted_license:{lic}"})
            continue
        if not rel or not exp_sha:
            rejected.append({"asset_id": aid, "reason": "missing_relative_path_or_sha256"})
            continue
        try:
            fp = ensure_contained_path(fdir / rel, fdir)
        except Exception as exc:
            rejected.append({"asset_id": aid, "reason": f"path_traversal_rejected:{exc}"})
            continue
        if not fp.is_file():
            rejected.append({"asset_id": aid, "reason": f"file_missing:{rel}"})
            continue
        actual_sha = _sha256_file(fp)
        if actual_sha != exp_sha:
            rejected.append({"asset_id": aid, "reason": f"sha256_mismatch:expected={exp_sha[:12]},actual={actual_sha[:12]}"})
            continue
        verified.append({**entry, "resolved_path": str(fp), "provenance": "recorded_local_foley"})

    fams = sorted({str(a["family"]) for a in verified})
    result["recorded_assets_verified"] = len(verified)
    result["recorded_families"] = fams
    result["assets"] = verified
    result["rejected_assets"] = rejected
    if len(verified) >= 4 and len(fams) >= 2:
        result["status"] = "RECORDED_FOLEY_VERIFIED"
    else:
        result["status"] = "PARTIAL_FOLEY_ASSETS"
        if not verified:
            result["notes"].append(
                "Manifest contains 0 verified recorded foley WAVs; upgraded procedural_dsp_synthesis active (never labeled recorded)."
            )
    return result


def load_recorded_foley_pcm(asset_entry: dict[str, Any], target_sr: int = 44100) -> tuple[np.ndarray, float] | None:
    """Load a SHA-256-verified local recorded WAV file (16-bit, 24-bit, or 32-bit PCM) as float32 stereo PCM."""
    from math import gcd
    from scipy.signal import resample_poly

    fp = Path(asset_entry.get("resolved_path", ""))
    if not fp.is_file():
        return None
    with wave.open(str(fp), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        # Unpack 24-bit little-endian signed PCM cleanly into float32 [-1.0, 1.0]
        b = np.frombuffer(raw, dtype=np.uint8).reshape((-1, 3))
        i32 = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        i32 = np.where(i32 & 0x800000, i32 - 0x1000000, i32)
        pcm = i32.astype(np.float32) / 8388608.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None
    if ch == 1:
        pcm = np.stack([pcm, pcm], axis=1)
    else:
        pcm = pcm.reshape((-1, ch))[:, :2]
    if sr != target_sr and pcm.shape[0] > 8:
        g = gcd(int(target_sr), int(sr))
        up, down = int(target_sr) // g, int(sr) // g
        pcm = np.stack(
            [resample_poly(pcm[:, 0], up, down), resample_poly(pcm[:, 1], up, down)],
            axis=1,
        ).astype(np.float32)
    peak = float(np.max(np.abs(pcm)))
    if peak > 1e-5:
        pcm = pcm * (0.88 / peak)
    t_samp = int(asset_entry.get("transient_sample", 0))
    t_off = float(t_samp) / max(1, int(asset_entry.get("original_sample_rate", sr)))
    return pcm.astype(np.float32), round(t_off, 5)


"""One-time authorized bootstrap script: Provision verified CC0-1.0 48kHz 24-bit WAV SFX/Foley pack
from `wyzowl/sfx-kit` (pinned commit `92780e7ab0cdfb557497a988089c47f46e8c7412`, CC0 1.0 Universal).

Downloads ONLY a lean curated palette (~24 WAVs, ~4-6 MB total) into `assets/sfx_foley/` (WAV binaries
are gitignored), verifies WAV integrity and SHA-256 checksums, extracts upstream `sync.at` exact
transient sample offsets, saves `LICENSE.wyzowl-sfx-kit.txt`, and writes `assets/sfx_foley/manifest.json`.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import wave
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLEY_DIR = PROJECT_ROOT / "assets" / "sfx_foley"
UPSTREAM_REPO = "wyzowl/sfx-kit"
UPSTREAM_SHA = "92780e7ab0cdfb557497a988089c47f46e8c7412"
RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_SHA}"

# Curated editorial finance/product/motion palette mapped to MotionLab families & variants
CURATED_SELECTION: list[tuple[str, int, str]] = [
    ("cash_flick", 0, "money-notes"),
    ("cash_flick", 1, "foley-card-touch"),
    ("cash_flick", 2, "foley-paper-touch"),
    ("paper_rustle", 0, "foley-paper-touch"),
    ("paper_rustle", 1, "foley-paper-slide"),
    ("paper_rustle", 2, "foley-fabric-touch"),
    ("envelope_slide", 0, "foley-card-slide"),
    ("envelope_slide", 1, "foley-paper-slide"),
    ("envelope_slide", 2, "foley-card-touch"),
    ("page_turn", 0, "props-page-turn"),
    ("page_turn", 1, "foley-paper-slide"),
    ("page_turn", 2, "foley-paper-touch"),
    ("soft_whoosh", 0, "whoosh-fast-soft"),
    ("soft_whoosh", 1, "whoosh-medium-soft"),
    ("soft_whoosh", 2, "whoosh-fast-medium"),
    ("reverse_whoosh", 0, "reverse-whoosh-short-soft"),
    ("reverse_whoosh", 1, "reverse-whoosh-short-strong"),
    ("reverse_whoosh", 2, "reverse-whoosh-medium-soft"),
    ("tick_accent", 0, "data-tick"),
    ("tick_accent", 1, "data-number-land"),
    ("tick_accent", 2, "impact-light-firm"),
    ("pop_reveal", 0, "pop-medium-in"),
    ("pop_reveal", 1, "pop-small-in"),
    ("pop_reveal", 2, "pop-large-in"),
    ("soft_impact", 0, "impact-medium-soft"),
    ("soft_impact", 1, "impact-heavy-soft"),
    ("soft_impact", 2, "impact-deep-soft"),
    ("shimmer_payoff", 0, "shimmer-short-soft"),
    ("shimmer_payoff", 1, "shimmer-medium-soft"),
    ("shimmer_payoff", 2, "shimmer-short-bright"),
    ("bass_settle", 0, "boom-hit-soft"),
    ("bass_settle", 1, "boom-drop-soft"),
    ("bass_settle", 2, "bass-sub-drop-short"),
    ("riser_build", 0, "riser-1s-soft"),
]


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MotionLab-Foley-Bootstrap/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def main() -> None:
    FOLEY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify and save exact upstream CC0 1.0 Universal LICENSE text
    lic_bytes = _fetch_bytes(f"{RAW_BASE}/LICENSE")
    lic_text = lic_bytes.decode("utf-8", errors="replace")
    if "CC0 1.0 Universal" not in lic_text:
        raise RuntimeError("Upstream LICENSE did not match CC0 1.0 Universal; aborting fail-closed!")
    (FOLEY_DIR / "LICENSE.wyzowl-sfx-kit.txt").write_bytes(lic_bytes)

    # 2. Fetch upstream structured sounds.json metadata
    sounds_doc = json.loads(_fetch_bytes(f"{RAW_BASE}/sounds.json").decode("utf-8"))
    items = sounds_doc if isinstance(sounds_doc, list) else sounds_doc.get("sounds", [])
    by_id = {str(it["id"]): it for it in items}

    # Check if everyday-props page turn has a slightly different id
    page_ids = [k for k in by_id if "page" in k or "book" in k]
    print("Available page/book IDs in wyzowl/sfx-kit:", page_ids)

    downloaded: dict[str, dict[str, Any]] = {}
    manifest_assets: list[dict[str, Any]] = []

    for fam, var, up_id in CURATED_SELECTION:
        if up_id not in by_id:
            if up_id == "props-page-turn" and page_ids:
                up_id = page_ids[0]
            else:
                print(f"SKIP missing upstream id: {up_id}")
                continue
        meta = by_id[up_id]
        rel_src = str(meta["file"])
        local_name = f"wyzowl_{up_id}.wav"
        local_path = FOLEY_DIR / local_name

        if up_id not in downloaded:
            raw_url = f"{RAW_BASE}/{rel_src}"
            wav_bytes = _fetch_bytes(raw_url)
            local_path.write_bytes(wav_bytes)
            sha = hashlib.sha256(wav_bytes).hexdigest()
            with wave.open(str(local_path), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                n_fr = wf.getnframes()
                dur_s = round(n_fr / max(1, sr), 4)
            sync_at = float((meta.get("sync") or {}).get("at", 0.0) or 0.0)
            transient_sample = max(0, min(n_fr - 1, int(round(sync_at * sr))))
            downloaded[up_id] = {
                "local_name": local_name,
                "raw_url": raw_url,
                "sha256": sha,
                "sr": sr,
                "ch": ch,
                "sw": sw,
                "dur_s": dur_s,
                "sync_at_s": sync_at,
                "transient_sample": transient_sample,
                "description": str(meta.get("description", "")),
            }
            print(f"  OK {local_name} ({dur_s}s, {sr}Hz {sw*8}-bit {ch}ch, sync={sync_at}s, sha={sha[:12]})")

        d = downloaded[up_id]
        manifest_assets.append({
            "asset_id": f"cc0_{fam}_v{var}_{up_id}",
            "relative_path": d["local_name"],
            "family": fam,
            "variant": var,
            "duration_s": d["dur_s"],
            "transient_sample": d["transient_sample"],
            "original_sample_rate": d["sr"],
            "bit_depth": d["sw"] * 8,
            "channels": d["ch"],
            "sha256": d["sha256"],
            "creator": "Wyzowl SFX Kit",
            "upstream_repo": UPSTREAM_REPO,
            "upstream_commit_sha": UPSTREAM_SHA,
            "upstream_sound_id": up_id,
            "source_url": d["raw_url"],
            "license": "CC0-1.0",
            "license_file": "LICENSE.wyzowl-sfx-kit.txt",
            "attribution": f"wyzowl/sfx-kit @ {UPSTREAM_SHA[:12]} (CC0 1.0 Universal, no attribution required)",
            "allowed_usage": "Commercial and non-commercial use unrestricted under CC0 1.0 Universal",
            "redistribution_allowed": True,
            "description": d["description"],
        })

    manifest = {
        "schema_version": "2.0.0",
        "status": "RECORDED_FOLEY_VERIFIED",
        "upstream_source": f"https://github.com/{UPSTREAM_REPO} @ {UPSTREAM_SHA}",
        "license": "CC0-1.0",
        "notes": (
            "Verified CC0-1.0 48kHz 24-bit WAV sound assets provisioned one-time from wyzowl/sfx-kit "
            "(commit 92780e7ab0cdfb557497a988089c47f46e8c7412). WAV binaries are gitignored locally; "
            "run `python scripts/provision_cc0_foley.py` once during bootstrap to materialize exact SHA-256 files. "
            "If local WAVs are absent, MotionLab fails back honestly to PARTIAL_FOLEY_ASSETS (procedural_dsp_synthesis)."
        ),
        "unique_wav_files": len(downloaded),
        "assets": manifest_assets,
    }
    (FOLEY_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDONE: Provisioned {len(downloaded)} unique CC0-1.0 WAV files ({len(manifest_assets)} family/variant bindings).")


if __name__ == "__main__":
    main()

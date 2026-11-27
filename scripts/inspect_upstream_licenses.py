"""Inspect HyperFrames local catalog IDs + fetch exact LICENSE headers & file trees of P0 Remotion repos."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".cache" / "repo_staging"


def fetch_raw(url: str, max_bytes: int = 16384) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MotionLab-R5-Survey/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read(max_bytes)
    except Exception:
        return None


def main() -> None:
    # 1. Exact HyperFrames catalog check
    data = json.loads((STAGE / "hf_catalog.json").read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("items") or data.get("blocks") or [])
    by_name = {str(it.get("name")): it for it in items}
    want_hf = [
        "apple-money-count", "count-up", "data-chart", "bar-chart-race",
        "kinetic-center-build", "kinetic-type-swap", "char-slam-explode", "split-flap-board",
        "weight-wave", "strikethrough-replace", "mk-callout-highlight", "flowchart", "flowchart-vertical",
        "mk-progress-stat", "mk-specs-list", "logo-outro", "organic-light-leak-overlay",
        "editorial-flash-overlay", "app-showcase", "exploded-stack", "hw-pipeline",
        "lt-mask-reveal", "lt-accent-underline", "lt-stack-bars", "freeze-frame-dressing",
    ]
    hf_verified = []
    print("=== VERIFIED HYPERFRAMES CATALOG BLOCKS ===")
    for w in want_hf:
        it = by_name.get(w)
        if it:
            t = str(it.get("type", ""))
            d = str(it.get("description", ""))
            hf_verified.append({"name": w, "type": t, "description": d, "tags": it.get("tags", [])})
            print(f"  FOUND  {w:28s} [{t:9s}] {d[:68]}")
        else:
            near = [k for k in by_name if w.split("-")[0] in k][:5]
            print(f"  MISS   {w:28s} near={near}")

    # 2. Check exact LICENSE text & hashes for all surveyed repos
    survey = json.loads((STAGE / "github_survey_raw.json").read_text(encoding="utf-8"))
    print("\n=== VERIFIED UPSTREAM LICENSES ===")
    for row in survey:
        sha = row.get("head_sha")
        repo = row["repo"]
        if not sha:
            row["license_verified"] = False
            continue
        lic_file = None
        lic_sha = None
        lic_head = None
        for fname in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
            raw = fetch_raw(f"https://raw.githubusercontent.com/{repo}/{sha}/{fname}", 32768)
            if raw:
                lic_file = fname
                lic_sha = hashlib.sha256(raw).hexdigest()
                lines = [ln.strip() for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
                lic_head = " | ".join(lines[:3])[:140]
                break
        row["license_file"] = lic_file
        row["license_file_sha256"] = lic_sha
        row["license_header"] = lic_head
        row["license_verified"] = bool(lic_file and lic_sha)
        print(f"  {row['code']:4s} {repo:42s} file={str(lic_file):11s} {str(lic_head)[:75]}")

    (STAGE / "github_survey_verified.json").write_text(json.dumps(survey, indent=2), encoding="utf-8")
    (STAGE / "hf_candidates_verified.json").write_text(json.dumps(hf_verified, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


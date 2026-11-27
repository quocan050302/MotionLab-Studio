"""Phase R2-A5 (ML-01.R5.R2): Verified Local Product Asset Catalog & Provenance Guard.

Manages owner-authorized product screenshots, demo recordings, workbook covers/pages, and logos
under `assets/product_catalog/manifest.json` with strict SHA-256, authorization, license,
and PII redaction checks. Never fabricates product UI when owner assets are absent: honestly
emits `ASSET_MISSING_CONCEPTUAL` and `PRODUCT_DEMO_PENDING`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from motionlab.config import PROJECT_ROOT
from motionlab.schemas import ensure_contained_path, sanitize_native

PRODUCT_CATALOG_DIR = PROJECT_ROOT / "assets" / "product_catalog"
PERMITTED_PRODUCT_LICENSES = {"OWNER_AUTHORIZED", "CC0-1.0", "COMMERCIAL_LICENSED", "MIT"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ProductAssetRecord:
    asset_id: str
    product_name: str
    asset_type: str  # "dashboard_screenshot" | "workbook_cover" | "workbook_page" | "product_demo_clip" | "logo"
    relative_path: str
    version: str
    owner_authorized: bool
    pii_redacted: bool
    license: str
    sha256: str
    width: int
    height: int
    supported_features: list[str]
    usage_constraints: str = ""
    verified: bool = False
    resolved_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def inspect_product_asset_catalog(catalog_dir: Path | None = None) -> dict[str, Any]:
    """Audit local product asset catalog with strict SHA-256, authorization, PII redaction, and path-containment checks."""
    cdir = (catalog_dir or PRODUCT_CATALOG_DIR).resolve()
    mpath = cdir / "manifest.json"
    result: dict[str, Any] = {
        "catalog_dir": str(cdir),
        "manifest_present": mpath.is_file(),
        "status": "ASSET_MISSING_CONCEPTUAL",
        "product_demo_status": "PRODUCT_DEMO_PENDING",
        "verified_assets": [],
        "rejected_assets": [],
        "operator_instructions": (
            "To enable real product demonstration instead of conceptual schematics, place owner-authorized, "
            "PII-redacted PNG/JPG/MP4 product assets inside assets/product_catalog/ and register exact "
            "SHA-256, owner_authorized=true, pii_redacted=true, and license in assets/product_catalog/manifest.json."
        ),
    }
    if not mpath.is_file():
        return result

    try:
        raw = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as exc:
        result["rejected_assets"].append({"asset_id": "manifest", "reason": f"invalid_json:{exc}"})
        return result

    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in raw.get("assets", []):
        aid = str(entry.get("asset_id", "unknown"))
        if not bool(entry.get("owner_authorized", False)):
            rejected.append({"asset_id": aid, "reason": "owner_authorization_missing"})
            continue
        if not bool(entry.get("pii_redacted", False)):
            rejected.append({"asset_id": aid, "reason": "pii_redaction_unconfirmed"})
            continue
        lic = str(entry.get("license", "")).upper()
        if lic not in PERMITTED_PRODUCT_LICENSES:
            rejected.append({"asset_id": aid, "reason": f"unpermitted_license:{lic}"})
            continue
        rel = str(entry.get("relative_path", ""))
        exp_sha = str(entry.get("sha256", ""))
        if not rel or not exp_sha:
            rejected.append({"asset_id": aid, "reason": "missing_path_or_sha256"})
            continue
        try:
            fp = ensure_contained_path(cdir / rel, cdir)
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
        verified.append({
            **entry,
            "authorized": True,
            "verified": True,
            "resolved_path": str(fp),
        })

    result["verified_assets"] = verified
    result["rejected_assets"] = rejected
    if verified:
        result["status"] = "AUTHORIZED_PRODUCT_ASSETS_VERIFIED"
        result["product_demo_status"] = "AUTHORIZED_DEMO_READY"
    return result


def match_product_assets_for_name(
    product_name: str | None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Match a spoken product name strictly against verified, owner-authorized catalog assets."""
    cat = catalog if catalog is not None else inspect_product_asset_catalog()
    if not product_name:
        return {"matched": False, "status": "NO_PRODUCT_SPOKEN", "assets": []}
    p_low = product_name.strip().lower()
    hits = [
        a for a in cat.get("verified_assets", [])
        if p_low in str(a.get("product_name", "")).lower() or str(a.get("product_name", "")).lower() in p_low
    ]
    if hits:
        return {
            "matched": True,
            "status": "AUTHORIZED_PRODUCT_ASSET_MATCHED",
            "product_demo_status": "AUTHORIZED_DEMO_ACTIVE",
            "assets": hits,
        }
    return {
        "matched": False,
        "status": "ASSET_MISSING_CONCEPTUAL",
        "product_demo_status": "PRODUCT_DEMO_PENDING",
        "assets": [],
    }

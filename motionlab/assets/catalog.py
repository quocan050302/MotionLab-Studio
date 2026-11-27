"""Phase A8 (ML-01.R5.R3): Strict Verified Local Product Asset Catalog, Decoded Media Check & Pixel Binding Guard.

Fixes D06:
  - Rejects empty/whitespace `product_id`/`product_name`/`asset_id`, disallowed extensions, and symlink/path traversal.
  - Decodes ACTUAL image/video bytes on disk (PIL / OpenCV) verifying non-corrupt pixels and real dimensions >= 64x64.
  - Strict exact normalized matching on `product_id`, `product_name`, or `approved_aliases` ONLY (ZERO substring matching;
    empty/whitespace query NEVER matches; ambiguous multi-product match fails closed!).
  - Never reports `AUTHORIZED_DEMO_ACTIVE` merely from catalog metadata: catalog reports `AUTHORIZED_ASSET_CANDIDATE_READY`
    until Remotion actually binds & renders the texture (`AUTHORIZED_DEMO_RENDERED_VERIFIED`).
  - Honestly emits `ASSET_MISSING_CONCEPTUAL` and `PRODUCT_DEMO_PENDING` when authorized owner assets are absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from motionlab.config import PROJECT_ROOT
from motionlab.schemas import ensure_contained_path, sanitize_native

PRODUCT_CATALOG_DIR = PROJECT_ROOT / "assets" / "product_catalog"
PERMITTED_PRODUCT_LICENSES = {"OWNER_AUTHORIZED", "CC0-1.0", "COMMERCIAL_LICENSED", "MIT"}
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".webm", ".mov"}
ALLOWED_ASSET_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _norm_name(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _decode_verify_media(fp: Path) -> tuple[bool, dict[str, Any]]:
    """Decode actual image or video bytes on disk to prove the asset is real, non-corrupt, and >=64x64."""
    ext = fp.suffix.lower()
    if ext not in ALLOWED_ASSET_EXTS:
        return False, {"reason": f"disallowed_extension:{ext}"}
    if ext in ALLOWED_IMAGE_EXTS:
        try:
            with Image.open(fp) as im:
                im.load()
                w, h = im.size
            if w < 64 or h < 64:
                return False, {"reason": f"image_too_small:{w}x{h}"}
            return True, {"media_kind": "image", "decoded_width": int(w), "decoded_height": int(h), "decoded_frames": 1}
        except Exception as exc:
            return False, {"reason": f"image_decode_failed:{exc}"}
    try:
        cap = cv2.VideoCapture(str(fp))
        ok, fr = cap.read()
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not ok or fr is None:
            return False, {"reason": "video_decode_failed_no_frame"}
        h, w = fr.shape[:2]
        if w < 64 or h < 64:
            return False, {"reason": f"video_too_small:{w}x{h}"}
        return True, {"media_kind": "video", "decoded_width": int(w), "decoded_height": int(h), "decoded_frames": max(1, nf)}
    except Exception as exc:
        return False, {"reason": f"video_decode_error:{exc}"}


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
    product_id: str = ""
    approved_aliases: list[str] = field(default_factory=list)
    usage_constraints: str = ""
    verified: bool = False
    decoded_verified: bool = False
    resolved_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def inspect_product_asset_catalog(catalog_dir: Path | None = None) -> dict[str, Any]:
    """Audit local product asset catalog with strict SHA-256, authorization, PII redaction, decode & path containment."""
    cdir = (catalog_dir or PRODUCT_CATALOG_DIR).resolve()
    mpath = cdir / "manifest.json"
    result: dict[str, Any] = {
        "schema_version": "3.0.0",
        "catalog_dir": str(cdir),
        "manifest_present": mpath.is_file(),
        "status": "ASSET_MISSING_CONCEPTUAL",
        "product_demo_status": "PRODUCT_DEMO_PENDING",
        "verified_assets": [],
        "rejected_assets": [],
        "operator_instructions": (
            "To enable real product demonstration instead of conceptual schematics, place owner-authorized, "
            "PII-redacted PNG/JPG/MP4 product assets inside assets/product_catalog/ and register exact "
            "SHA-256, non-empty product_name/product_id, owner_authorized=true, pii_redacted=true, and license in manifest.json."
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
        aid = str(entry.get("asset_id", "")).strip()
        pname = str(entry.get("product_name", "")).strip()
        pid = str(entry.get("product_id", "")).strip() or _norm_name(pname).replace(" ", "_")
        if not aid:
            rejected.append({"asset_id": "missing", "reason": "empty_asset_id"})
            continue
        if not pname:
            rejected.append({"asset_id": aid, "reason": "empty_product_name"})
            continue
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
        rel = str(entry.get("relative_path", "")).strip()
        exp_sha = str(entry.get("sha256", "")).strip()
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
        dec_ok, dec_info = _decode_verify_media(fp)
        if not dec_ok:
            rejected.append({"asset_id": aid, "reason": dec_info.get("reason", "media_decode_failed")})
            continue

        # Filter empty aliases strictly
        raw_aliases = entry.get("approved_aliases") or []
        clean_aliases = [str(x).strip() for x in raw_aliases if isinstance(x, str) and str(x).strip()]
        verified.append({
            **entry,
            "asset_id": aid,
            "product_id": pid,
            "product_name": pname,
            "approved_aliases": clean_aliases,
            "authorized": True,
            "verified": True,
            "decoded_verified": True,
            "media_kind": dec_info["media_kind"],
            "decoded_width": dec_info["decoded_width"],
            "decoded_height": dec_info["decoded_height"],
            "decoded_frames": dec_info["decoded_frames"],
            "resolved_path": str(fp),
        })

    result["verified_assets"] = verified
    result["rejected_assets"] = rejected
    if verified:
        result["status"] = "AUTHORIZED_PRODUCT_ASSETS_VERIFIED"
        result["product_demo_status"] = "AUTHORIZED_ASSET_CANDIDATE_READY"
    return result


def match_product_assets_for_name(
    product_name: str | None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fixes D06: Strict exact normalized match on product_name / product_id / approved_aliases (ZERO substring matching!)."""
    cat = catalog if catalog is not None else inspect_product_asset_catalog()
    target = _norm_name(product_name)
    if not target:
        return {
            "matched": False,
            "status": "NO_PRODUCT_SPOKEN",
            "product_demo_status": "PRODUCT_DEMO_PENDING",
            "assets": [],
        }

    hits: list[dict[str, Any]] = []
    matched_product_ids: set[str] = set()
    for a in cat.get("verified_assets", []):
        if not (a.get("verified") and a.get("authorized") and a.get("decoded_verified", True)):
            continue
        pname = _norm_name(a.get("product_name"))
        pid = _norm_name(a.get("product_id"))
        aliases = {_norm_name(x) for x in (a.get("approved_aliases") or []) if _norm_name(x)}
        if target == pname or (pid and target == pid) or target in aliases:
            hits.append(a)
            matched_product_ids.add(pid or pname)

    if len(matched_product_ids) > 1:
        return {
            "matched": False,
            "status": "AMBIGUOUS_PRODUCT_MATCH",
            "product_demo_status": "PRODUCT_DEMO_PENDING",
            "ambiguous_product_ids": sorted(matched_product_ids),
            "assets": [],
        }

    if hits:
        return {
            "matched": True,
            "status": "AUTHORIZED_PRODUCT_ASSET_MATCHED",
            "product_demo_status": "AUTHORIZED_ASSET_CANDIDATE_READY",
            "assets": hits,
        }
    return {
        "matched": False,
        "status": "ASSET_MISSING_CONCEPTUAL",
        "product_demo_status": "PRODUCT_DEMO_PENDING",
        "assets": [],
    }

"""Phase R2-A5 / R2-A8 (ML-01.R5.R2): Unit & Negative Tests for Product Asset Catalog & Independent Visual Critic."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motionlab.assets.catalog import inspect_product_asset_catalog, match_product_assets_for_name
from motionlab.composition.visual_critic import inspect_local_vlm


class TestR5R2ProductCatalogAndCritic(unittest.TestCase):
    def test_missing_catalog_returns_honest_conceptual_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cat = inspect_product_asset_catalog(Path(td))
            self.assertEqual(cat["status"], "ASSET_MISSING_CONCEPTUAL")
            self.assertEqual(cat["product_demo_status"], "PRODUCT_DEMO_PENDING")
            m = match_product_assets_for_name("The Next Paycheck Plan", catalog=cat)
            self.assertFalse(m["matched"])
            self.assertEqual(m["status"], "ASSET_MISSING_CONCEPTUAL")

    def test_unauthorized_unredacted_tampered_and_traversal_assets_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td)
            ok_img = cdir / "demo.png"
            from PIL import Image
            Image.new("RGB", (128, 128), (36, 59, 50)).save(ok_img, "PNG")
            ok_sha = hashlib.sha256(ok_img.read_bytes()).hexdigest()

            manifest = {
                "assets": [
                    {
                        "asset_id": "ok_dash",
                        "product_name": "Clarity Ledger",
                        "asset_type": "dashboard_screenshot",
                        "relative_path": "demo.png",
                        "version": "1.0",
                        "owner_authorized": True,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": ok_sha,
                    },
                    {
                        "asset_id": "bad_unauth",
                        "product_name": "Clarity Ledger",
                        "relative_path": "demo.png",
                        "owner_authorized": False,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": ok_sha,
                    },
                    {
                        "asset_id": "bad_pii",
                        "product_name": "Clarity Ledger",
                        "relative_path": "demo.png",
                        "owner_authorized": True,
                        "pii_redacted": False,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": ok_sha,
                    },
                    {
                        "asset_id": "bad_sha",
                        "product_name": "Clarity Ledger",
                        "relative_path": "demo.png",
                        "owner_authorized": True,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": "f" * 64,
                    },
                    {
                        "asset_id": "bad_trav",
                        "product_name": "Clarity Ledger",
                        "relative_path": "../outside.png",
                        "owner_authorized": True,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": ok_sha,
                    },
                ]
            }
            (cdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            cat = inspect_product_asset_catalog(cdir)
            self.assertEqual(cat["status"], "AUTHORIZED_PRODUCT_ASSETS_VERIFIED")
            self.assertEqual(len(cat["verified_assets"]), 1)
            self.assertEqual(cat["verified_assets"][0]["asset_id"], "ok_dash")
            self.assertEqual(len(cat["rejected_assets"]), 4)

            hit = match_product_assets_for_name("Clarity Ledger", catalog=cat)
            self.assertTrue(hit["matched"])
            miss = match_product_assets_for_name("Unrelated Product", catalog=cat)
            self.assertFalse(miss["matched"])

    def test_vlm_critic_honestly_reports_not_run_without_local_vlm(self) -> None:
        vlm = inspect_local_vlm()
        self.assertFalse(vlm["available"])
        self.assertEqual(vlm["status"], "VISION_CRITIC_NOT_RUN")


if __name__ == "__main__":
    unittest.main()

"""Phase A8 (ML-01.R5.R3): Product Grounding Truth, Decoded Bytes Check & Zero Runtime Olivia Hardcodes (D05, D06)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from motionlab.assets.catalog import inspect_product_asset_catalog, match_product_assets_for_name
from motionlab.composition.primitive_adapters import build_adapted_hyperframes_html
from motionlab.config import PROJECT_ROOT


class TestR5R3ProductGrounding(unittest.TestCase):
    def test_d05_zero_olivia_hardcodes_in_runtime_adapters_and_tsx(self) -> None:
        """P0 Fix D05: Neither Remotion TSX nor HyperFrames adapters may contain Olivia default fallback strings."""
        tsx_src = (PROJECT_ROOT / "renderers" / "remotion" / "src" / "GenerativeSceneComposition.tsx").read_text(encoding="utf-8")
        for forbidden in ("The Next Paycheck Plan", "56-page workbook", "Notion dashboard", "01 / 56", "$1,200 TO SPEND", "Give every dollar a purpose"):
            self.assertNotIn(forbidden, tsx_src, f"Olivia hardcoded fallback still present in TSX: {forbidden}")

        # Calling every HF adapter with empty parameters must NEVER leak Olivia product/amounts!
        for pid in ("hf_apple_money_count", "hf_mk_progress_stat", "hf_mk_specs_list", "hf_mk_callout_highlight", "hf_lt_mask_reveal", "hf_strikethrough_replace", "hf_data_chart", "hf_exploded_stack"):
            html = build_adapted_hyperframes_html(pid, 720, 1280, 24.0, 2.5, {})
            for leaked in ("The Next Paycheck Plan", "56-page", "Notion dashboard", "$1,200", "Give every dollar"):
                self.assertNotIn(leaked, html, f"Olivia string '{leaked}' leaked from empty-param adapter {pid}")

    def test_d06_strict_exact_matching_corrupt_bytes_and_ambiguity_guards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td)
            # 1. Valid decoded 200x200 PNG
            real_png = cdir / "atlas_cover.png"
            Image.new("RGB", (200, 200), (36, 59, 50)).save(real_png, "PNG")
            real_sha = hashlib.sha256(real_png.read_bytes()).hexdigest()

            # 2. Corrupt fake PNG bytes (MUST BE REJECTED!)
            fake_png = cdir / "corrupt.png"
            fake_png.write_bytes(b"\x89PNG\r\n\x1a\nCORRUPT_NOT_AN_IMAGE")
            fake_sha = hashlib.sha256(fake_png.read_bytes()).hexdigest()

            # 3. Disallowed extension (.txt)
            bad_ext = cdir / "notes.txt"
            bad_ext.write_text("hello", encoding="utf-8")
            bad_ext_sha = hashlib.sha256(bad_ext.read_bytes()).hexdigest()

            manifest = {
                "assets": [
                    {
                        "asset_id": "atlas_01",
                        "product_id": "budget_atlas",
                        "product_name": "Budget Atlas",
                        "approved_aliases": ["Atlas Planner"],
                        "asset_type": "workbook_cover",
                        "relative_path": "atlas_cover.png",
                        "version": "1.0",
                        "owner_authorized": True,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": real_sha,
                    },
                    {
                        "asset_id": "corrupt_01",
                        "product_name": "Corrupt Product",
                        "asset_type": "workbook_cover",
                        "relative_path": "corrupt.png",
                        "owner_authorized": True,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": fake_sha,
                    },
                    {
                        "asset_id": "badext_01",
                        "product_name": "Text Product",
                        "asset_type": "workbook_cover",
                        "relative_path": "notes.txt",
                        "owner_authorized": True,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": bad_ext_sha,
                    },
                    {
                        "asset_id": "empty_name_01",
                        "product_name": "   ",
                        "asset_type": "workbook_cover",
                        "relative_path": "atlas_cover.png",
                        "owner_authorized": True,
                        "pii_redacted": True,
                        "license": "OWNER_AUTHORIZED",
                        "sha256": real_sha,
                    },
                ]
            }
            (cdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            cat = inspect_product_asset_catalog(cdir)
            self.assertEqual(cat["status"], "AUTHORIZED_PRODUCT_ASSETS_VERIFIED")
            self.assertEqual(cat["product_demo_status"], "AUTHORIZED_ASSET_CANDIDATE_READY")
            self.assertEqual(len(cat["verified_assets"]), 1)
            self.assertTrue(cat["verified_assets"][0]["decoded_verified"])
            self.assertEqual(cat["verified_assets"][0]["decoded_width"], 200)
            rej_ids = {r["asset_id"] for r in cat["rejected_assets"]}
            self.assertEqual(rej_ids, {"corrupt_01", "badext_01", "empty_name_01"})

            # Empty / whitespace / partial substring MUST NOT match!
            self.assertFalse(match_product_assets_for_name("", catalog=cat)["matched"])
            self.assertFalse(match_product_assets_for_name("   ", catalog=cat)["matched"])
            self.assertFalse(match_product_assets_for_name("Atlas", catalog=cat)["matched"])
            self.assertFalse(match_product_assets_for_name("Budget", catalog=cat)["matched"])

            # Exact name or approved alias matches cleanly (status CANDIDATE_READY until rendered!)
            hit1 = match_product_assets_for_name("Budget Atlas", catalog=cat)
            self.assertTrue(hit1["matched"])
            self.assertEqual(hit1["product_demo_status"], "AUTHORIZED_ASSET_CANDIDATE_READY")
            hit2 = match_product_assets_for_name("atlas planner", catalog=cat)
            self.assertTrue(hit2["matched"])

            # Ambiguous alias collision across two products MUST fail closed!
            ambig_cat = {
                "verified_assets": [
                    {"asset_id": "a1", "product_id": "p1", "product_name": "Plan A", "approved_aliases": ["Shared"], "authorized": True, "verified": True, "decoded_verified": True},
                    {"asset_id": "a2", "product_id": "p2", "product_name": "Plan B", "approved_aliases": ["Shared"], "authorized": True, "verified": True, "decoded_verified": True},
                ]
            }
            amb = match_product_assets_for_name("Shared", catalog=ambig_cat)
            self.assertFalse(amb["matched"])
            self.assertEqual(amb["status"], "AMBIGUOUS_PRODUCT_MATCH")


if __name__ == "__main__":
    unittest.main()

"""ML-01.R5.R5 Unit, Negative & Mutation Tests — Part 2:
  - LocalModelRouter & Qwen3-VL-2B VLM manifest verification (A2)
  - DirectorToolRegistry prompt-injection / traversal / budget guards (A4)
  - Multi-direction causal ranking sensitivity & filename invariance (A4, D12)
  - Claim-to-visual coverage matrix ensuring TEXT_ONLY / ILLUSTRATION never forge ACTUAL_PRODUCT (A5, A9)
  - Multi-niche & 16:9 landscape dynamic layout solver (zero hardcoded y=1255!) & 3 distinct hook candidates (A6, A7, A10)
  - Multi-corner static watermark detector synthetic positive vs negative clip (A3)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from motionlab.assets.catalog import inspect_product_asset_catalog, match_product_assets_for_name
from motionlab.composition.director_pipeline import _build_scene_graph_v2_from_claims
from motionlab.composition.multi_source_timeline import MultiSourceEditMap
from motionlab.composition.scene_graph_v2 import (
    build_three_hook_candidates,
    compile_scene_v2_html,
    solve_dynamic_safe_layout,
)
from motionlab.semantic.creative_director import (
    DirectorToolRegistry,
    build_claim_visual_coverage_matrix,
    build_intent_brief,
    extract_claims_and_questions,
    propose_global_creative_directions,
)
from motionlab.semantic.model_router import LocalModelRouter, inspect_vlm_model
from motionlab.semantic.video_evidence import detect_static_corner_watermarks


class TestR5R5DirectorVlmAndSceneGraph(unittest.TestCase):
    def test_vlm_spec_registered_and_model_router_rejects_unknown_model(self) -> None:
        """A2: Local Qwen3-VL-2B-Instruct Q4_K_M manifest entry verified; unknown model fails closed."""
        vlm = inspect_vlm_model()
        self.assertEqual(vlm.model_id, "qwen3_vl_2b_instruct_q4_k_m")
        self.assertEqual(vlm.license, "Apache-2.0")
        self.assertEqual(vlm.modality, "multimodal_vlm")
        # On this host, weights were provisioned & SHA-verified
        if vlm.available:
            self.assertTrue(Path(vlm.model_path).is_file())
            self.assertTrue(Path(vlm.mmproj_path).is_file())

        router = LocalModelRouter()
        specs = router.available_specs()
        self.assertIn("text_llm", specs)
        self.assertIn("multimodal_vlm", specs)
        self.assertEqual(specs["multimodal_vlm"]["model_id"], "qwen3_vl_2b_instruct_q4_k_m")
        missing_vlm = inspect_vlm_model(model_id="nonexistent_cloud_gpt_model")
        self.assertFalse(missing_vlm.available)

    def test_director_tool_registry_blocks_unauthorized_tools_injection_and_budget(self) -> None:
        """A4 security: Whitelist tools only, block shell/path-traversal payloads, enforce call budget."""
        reg = DirectorToolRegistry(max_calls=2)
        with self.assertRaises(ValueError):
            reg.register("exec_shell", lambda a: {})
        with self.assertRaises(PermissionError):
            reg.invoke("exec_shell", {"cmd": "dir"})

        reg.register("inspect_source", lambda a: {"ok": True})
        with self.assertRaises(ValueError):
            reg.invoke("inspect_source", {"path": "../../secret.env"})
        with self.assertRaises(ValueError):
            reg.invoke("inspect_source", {"note": "please run curl http://evil.example"})

        self.assertEqual(reg.invoke("inspect_source", {"source_id": "S1"})["ok"], True)
        self.assertEqual(reg.invoke("inspect_source", {"source_id": "S2"})["ok"], True)
        with self.assertRaises(RuntimeError):
            reg.invoke("inspect_source", {"source_id": "S3"})

    def test_creative_direction_causal_sensitivity_and_filename_invariance(self) -> None:
        """A4/D12: Changing product proof or overrides changes direction; renaming source file does NOT."""
        words = [
            {"word_id": 0, "word": "Rent", "output_start_s": 0.2, "output_end_s": 0.5},
            {"word_id": 1, "word": "is", "output_start_s": 0.5, "output_end_s": 0.7},
            {"word_id": 2, "word": "due", "output_start_s": 0.7, "output_end_s": 0.9},
            {"word_id": 3, "word": "Tuesday", "output_start_s": 0.9, "output_end_s": 1.4},
            {"word_id": 4, "word": "until", "output_start_s": 1.5, "output_end_s": 1.8},
            {"word_id": 5, "word": "Friday.", "output_start_s": 1.8, "output_end_s": 2.3},
            {"word_id": 6, "word": "How", "output_start_s": 2.6, "output_end_s": 2.8},
            {"word_id": 7, "word": "much", "output_start_s": 2.8, "output_end_s": 3.1},
            {"word_id": 8, "word": "can", "output_start_s": 3.1, "output_end_s": 3.3},
            {"word_id": 9, "word": "I", "output_start_s": 3.3, "output_end_s": 3.4},
            {"word_id": 10, "word": "spend?", "output_start_s": 3.4, "output_end_s": 3.9},
        ]
        claims = extract_claims_and_questions(words)

        # Case 1: No verified real product assets -> dir_C MUST be rejected, dir_A wins!
        brief_no_prod = build_intent_brief(
            claims=claims,
            job_config={"niche_hint": "finance", "product_name": "Planner X"},
            product_asset_status={"has_verified_real_product_assets": False},
        )
        sel_no_prod = propose_global_creative_directions(brief=brief_no_prod, claims=claims, model_router=None)
        self.assertEqual(sel_no_prod["selected_direction_id"], "dir_A_editorial_visual_metaphor")
        dir_c = next(d for d in sel_no_prod["directions_considered"] if d["direction_id"] == "dir_C_product_led_demo")
        self.assertFalse(dir_c["feasible"])

        # Case 2: Verified real product assets available -> dir_C becomes feasible & wins!
        brief_prod = build_intent_brief(
            claims=claims,
            job_config={"niche_hint": "finance", "product_name": "Planner X"},
            product_asset_status={"has_verified_real_product_assets": True},
        )
        sel_prod = propose_global_creative_directions(brief=brief_prod, claims=claims, model_router=None)
        self.assertEqual(sel_prod["selected_direction_id"], "dir_C_product_led_demo")

        # Case 3: Candidate score override shifts to dir_B causally!
        sel_override = propose_global_creative_directions(
            brief=brief_no_prod,
            claims=claims,
            model_router=None,
            candidate_score_overrides={"dir_B_authentic_vlog_spatial": 0.50},
        )
        self.assertEqual(sel_override["selected_direction_id"], "dir_B_authentic_vlog_spatial")

        # Case 4: Filename change alone produces identical selection!
        brief_a = build_intent_brief(claims=claims, job_config={"niche_hint": "tutorial", "job_id": "olivia_v6"}, product_asset_status={})
        brief_b = build_intent_brief(claims=claims, job_config={"niche_hint": "tutorial", "job_id": "unrelated_999"}, product_asset_status={})
        self.assertEqual(
            propose_global_creative_directions(brief=brief_a, claims=claims)["selected_direction_id"],
            propose_global_creative_directions(brief=brief_b, claims=claims)["selected_direction_id"],
        )

    def test_claim_visual_coverage_matrix_never_forges_product_proof_from_text_or_illustration(self) -> None:
        """A5/A9 negative test: TEXT_ONLY and ILLUSTRATION never satisfy ACTUAL_PRODUCT proof."""
        claims = [
            {"claim_id": "c0", "role": "system_mechanism_benefits", "text": "It shows what is reserved.", "start_s": 1.0, "end_s": 4.0, "requires_visual_evidence": True},
        ]
        scenes_text = [{"scene_id": "s0", "treatment_mode": "spatial_editorial_anchor", "start_s": 1.0, "end_s": 4.0, "start_frame": 30, "end_frame_exclusive": 120}]
        cov_text = build_claim_visual_coverage_matrix(claims=claims, scenes=scenes_text, product_proof_available=False)
        self.assertFalse(cov_text["actual_product_proof_verified"])
        self.assertEqual(cov_text["rows"][0]["evidence_class"], "TEXT_ONLY")

        scenes_ill = [{"scene_id": "s1", "treatment_mode": "illustrative_system_board", "start_s": 1.0, "end_s": 4.0, "start_frame": 30, "end_frame_exclusive": 120}]
        cov_ill = build_claim_visual_coverage_matrix(claims=claims, scenes=scenes_ill, product_proof_available=False)
        self.assertFalse(cov_ill["actual_product_proof_verified"])
        self.assertEqual(cov_ill["rows"][0]["evidence_class"], "ILLUSTRATION")

        scenes_real = [{"scene_id": "s2", "treatment_mode": "actual_product_capture", "start_s": 1.0, "end_s": 4.0, "start_frame": 30, "end_frame_exclusive": 120}]
        cov_real = build_claim_visual_coverage_matrix(claims=claims, scenes=scenes_real, product_proof_available=True)
        self.assertTrue(cov_real["actual_product_proof_verified"])
        self.assertEqual(cov_real["rows"][0]["evidence_class"], "ACTUAL_PRODUCT")

    def test_dynamic_layout_solver_adapts_9x16_and_16x9_and_builds_three_distinct_hooks(self) -> None:
        """A6/A7/A10: Dynamic layout avoids face bounds on both 9:16 and 16:9; 3 hooks have distinct modes."""
        lay_916 = solve_dynamic_safe_layout(
            width=1080, height=1920, treatment_mode="calendar_timeline_gap",
            face_bounds={"x_min": 0.25, "y_min": 0.20, "x_max": 0.72, "y_max": 0.52},
        )
        lay_169 = solve_dynamic_safe_layout(
            width=1920, height=1080, treatment_mode="calendar_timeline_gap",
            face_bounds={"x_min": 0.35, "y_min": 0.12, "x_max": 0.65, "y_max": 0.55},
        )
        self.assertFalse(lay_916["overlaps_face"])
        self.assertFalse(lay_169["overlaps_face"])
        self.assertNotEqual(lay_916["y_px"], lay_169["y_px"])
        self.assertLess(lay_169["y_px"], 1080)

        # 3 materially distinct hook candidates
        hooks = build_three_hook_candidates(
            hook_claim={"start_s": 0.1, "end_s": 3.4, "text": "Why does my code fail after deploy?"},
            next_claim={"weekdays_mentioned": ["Monday", "Thursday"]},
            width=1080, height=1920, fps=30,
            face_bounds={"x_min": 0.25, "y_min": 0.20, "x_max": 0.72, "y_max": 0.52},
            safe_zones=None,
        )
        self.assertEqual(len(hooks), 3)
        modes = [h["scene"]["treatment_mode"] for h in hooks]
        self.assertEqual(len(set(modes)), 3)

        # Every mode compiles valid distinct HTML with GSAP timeline
        for m in ("spatial_editorial_anchor", "calendar_timeline_gap", "split_dilemma_metaphor", "motivated_object_bridge", "illustrative_system_board", "minimal_cta_anchor"):
            html_s, transp = compile_scene_v2_html({"scene_id": f"sc_{m}", "treatment_mode": m, "duration_s": 2.4, "params": {"headline": "Test"}})
            self.assertTrue(transp)
            self.assertIn("gsap.timeline", html_s)

    def test_multi_corner_static_watermark_detector_positive_and_negative_clips(self) -> None:
        """A3: Detects synthetic static corner sparkle mark; clean animated clip returns False."""
        with tempfile.TemporaryDirectory() as td:
            p_mark = Path(td) / "with_mark.mp4"
            p_clean = Path(td) / "clean.mp4"
            w, h = 270, 480
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw1 = cv2.VideoWriter(str(p_mark), fourcc, 15.0, (w, h))
            vw2 = cv2.VideoWriter(str(p_clean), fourcc, 15.0, (w, h))
            for f in range(30):
                bg = np.full((h, w, 3), (75 + (f % 10), 80 + (f % 8), 85), dtype=np.uint8)
                # Moving stripe so background isn't identical everywhere
                cv2.rectangle(bg, ((f * 9) % (w - 30), 60), (((f * 9) % (w - 30)) + 28, 300), (140, 130, 120), -1)
                vw2.write(bg)
                bg_m = bg.copy()
                # Static crisp 4-point sparkle in bottom-right corner
                cx, cy = int(w * 0.88), int(h * 0.91)
                pts = np.array([[cx, cy - 12], [cx + 4, cy - 4], [cx + 12, cy], [cx + 4, cy + 4], [cx, cy + 12], [cx - 4, cy + 4], [cx - 12, cy], [cx - 4, cy - 4]])
                cv2.fillPoly(bg_m, [pts], (242, 242, 242))
                vw1.write(bg_m)
            vw1.release()
            vw2.release()

            res_m = detect_static_corner_watermarks(p_mark, num_samples=10)
            res_c = detect_static_corner_watermarks(p_clean, num_samples=10)
            self.assertTrue(res_m["detected_bright_static_mark"])
            self.assertFalse(res_c["detected_bright_static_mark"])

    def test_actual_product_capture_html_embeds_local_assets_and_blocks_traversal(self) -> None:
        """A6/D05: actual_product_capture compiles verified local assets cleanly and rejects unsafe URLs/traversal."""
        html_s, transp = compile_scene_v2_html({
            "scene_id": "sc_prod",
            "treatment_mode": "actual_product_capture",
            "duration_s": 4.8,
            "params": {
                "title": "Demo Plan",
                "provenance_badge": "REAL WORKBOOK · FICTIONAL EXAMPLE",
                "pages": [
                    {"asset_key": "prod_00_p07.png", "step_pill": "01 BILLS", "caption_kicker": "STEP 1", "caption_title": "Reserved First", "trigger_s": 0.1},
                    {"asset_key": "prod_01_p10.png", "step_pill": "02 GAP", "caption_kicker": "STEP 2", "caption_title": "Payday Timeline", "trigger_s": 1.8},
                ],
            },
        })
        self.assertTrue(transp)
        self.assertIn("./prod_00_p07.png", html_s)
        self.assertIn("./prod_01_p10.png", html_s)
        self.assertIn("REAL WORKBOOK", html_s)
        self.assertNotIn("http://", html_s.lower())
        self.assertNotIn("https://", html_s.lower())

        # Bridge & CTA with local cover asset key
        br_html, _ = compile_scene_v2_html({
            "scene_id": "sc_br",
            "treatment_mode": "motivated_object_bridge",
            "duration_s": 1.2,
            "params": {"title": "Demo Plan", "featured_asset_key": "prod_cover.png"},
        })
        self.assertIn("./prod_cover.png", br_html)
        cta_html, _ = compile_scene_v2_html({
            "scene_id": "sc_cta",
            "treatment_mode": "minimal_cta_anchor",
            "duration_s": 2.0,
            "params": {"headline": "Link in caption", "cover_asset_key": "cta_cover.png"},
        })
        self.assertIn("./cta_cover.png", cta_html)

        # Fail-closed negative checks: empty pages, traversal, remote URL, data URI
        with self.assertRaises(ValueError):
            compile_scene_v2_html({"scene_id": "bad0", "treatment_mode": "actual_product_capture", "duration_s": 2.0, "params": {"pages": []}})
        for bad_key in ("../escape.png", "https://evil.example/a.png", "data:image/png;base64,AAA", "sub/dir.png"):
            with self.assertRaises(ValueError):
                compile_scene_v2_html({
                    "scene_id": "bad1",
                    "treatment_mode": "actual_product_capture",
                    "duration_s": 2.0,
                    "params": {"pages": [{"asset_key": bad_key, "caption_title": "Bad"}]},
                })

    def test_director_catalog_binding_switches_causally_to_actual_product_capture(self) -> None:
        """D05 end-to-end binding: verified AssetCatalog assets causally drive dir_C -> actual_product_capture."""
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td)
            img_specs = [
                ("demo_cover", "workbook_cover", "cover.png", ["cover"]),
                ("demo_p1", "workbook_page", "p1.png", ["already_spoken_for"]),
                ("demo_p2", "workbook_page", "p2.png", ["payday_timeline"]),
                ("demo_p3", "workbook_page", "p3.png", ["actually_spend"]),
            ]
            manifest_assets = []
            for aid, atype, fn, feats in img_specs:
                fp = cdir / fn
                Image.new("RGB", (320, 460), (22, 56, 46)).save(fp, "PNG")
                manifest_assets.append({
                    "asset_id": aid,
                    "product_id": "demo_plan",
                    "product_name": "Demo Plan",
                    "asset_type": atype,
                    "relative_path": fn,
                    "version": "1.0",
                    "owner_authorized": True,
                    "pii_redacted": True,
                    "license": "OWNER_AUTHORIZED",
                    "sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
                    "width": 320,
                    "height": 460,
                    "supported_features": feats,
                })
            (cdir / "manifest.json").write_text(json.dumps({"schema_version": "3.0.0", "assets": manifest_assets}), encoding="utf-8")

            cat = inspect_product_asset_catalog(cdir)
            self.assertEqual(cat["status"], "AUTHORIZED_PRODUCT_ASSETS_VERIFIED")
            match = match_product_assets_for_name("Demo Plan", catalog=cat)
            self.assertTrue(match["matched"])
            self.assertEqual(len(match["assets"]), 4)

            words = [
                {"word_id": 0, "word": "Check", "output_start_s": 0.1, "output_end_s": 0.4},
                {"word_id": 1, "word": "this.", "output_start_s": 0.4, "output_end_s": 0.8},
                {"word_id": 2, "word": "It", "output_start_s": 2.0, "output_end_s": 2.2},
                {"word_id": 3, "word": "shows", "output_start_s": 2.2, "output_end_s": 2.6},
                {"word_id": 4, "word": "you", "output_start_s": 2.6, "output_end_s": 2.9},
                {"word_id": 5, "word": "what's", "output_start_s": 2.9, "output_end_s": 3.3},
                {"word_id": 6, "word": "spoken", "output_start_s": 3.3, "output_end_s": 3.8},
                {"word_id": 7, "word": "for,", "output_start_s": 3.8, "output_end_s": 4.2},
                {"word_id": 8, "word": "what", "output_start_s": 4.3, "output_end_s": 4.6},
                {"word_id": 9, "word": "needs", "output_start_s": 4.6, "output_end_s": 5.0},
                {"word_id": 10, "word": "payday,", "output_start_s": 5.0, "output_end_s": 5.6},
                {"word_id": 11, "word": "and", "output_start_s": 5.7, "output_end_s": 5.9},
                {"word_id": 12, "word": "spend.", "output_start_s": 5.9, "output_end_s": 6.6},
                {"word_id": 13, "word": "Link", "output_start_s": 7.0, "output_end_s": 7.3},
                {"word_id": 14, "word": "in", "output_start_s": 7.3, "output_end_s": 7.5},
                {"word_id": 15, "word": "caption.", "output_start_s": 7.5, "output_end_s": 8.2},
            ]
            claims = extract_claims_and_questions(words)
            brief = build_intent_brief(
                claims=claims,
                job_config={"niche_hint": "finance", "product_name": "Demo Plan"},
                product_asset_status={"has_verified_real_product_assets": True},
            )
            sel = propose_global_creative_directions(brief=brief, claims=claims, model_router=None)
            self.assertEqual(sel["selected_direction_id"], "dir_C_product_led_demo")

            dummy_map = MultiSourceEditMap(
                schema_version="1.0.0",
                sources=[],
                target_width=1080,
                target_height=1920,
                fps_num=30,
                fps_den=1,
                sample_rate=48000,
                channels=2,
                output_total_frames=270,
                output_duration_s=9.0,
                output_total_samples=432000,
                kept_spans=[],
                cuts=[],
                map_hash="dummy",
            )
            scenes = _build_scene_graph_v2_from_claims(
                claims=claims,
                selected_direction=sel["selected_direction"],
                winning_hook_scene=None,
                brief=brief,
                edit_map=dummy_map,
                face_bounds=None,
                bridge_span=None,
                verified_product_assets=match["assets"],
                edited_words=words,
            )
            prod_scenes = [s for s in scenes if s["treatment_mode"] == "actual_product_capture"]
            self.assertEqual(len(prod_scenes), 1)
            self.assertEqual(len(prod_scenes[0]["extra_assets"]), 3)
            self.assertEqual(prod_scenes[0]["asset_ids"], ["demo_p1", "demo_p2", "demo_p3"])
            cov = build_claim_visual_coverage_matrix(claims=claims, scenes=scenes, product_proof_available=True)
            self.assertTrue(cov["actual_product_proof_verified"])


if __name__ == "__main__":
    unittest.main()

"""ML-01.R5.R6 Comprehensive Unit, Negative, Mutation, Causal & 14-Niche Generalization Tests.

Covers:
  - A1: Source dialogue completeness vs retained speech preservation + critical token drop mutations
  - A2/A3: Generic atomic claim splitting (non-finance!), relevance-threshold asset rejection, causal model-driven plans
  - A4: Zero-AGPL PDF region cropper fail-closed guards (tampered SHA, blank crop, tiny/OOB bbox) & feature-level proof ledger
  - A5/A6/11A: SceneGraph V3 14-niche universal transfer (zero leaked Olivia/finance strings!) & 16:9 landscape layout
  - A7: Blind permutation-invariant hook competition & pixel-swap causality (zero mode/id bonuses!)
  - A8/D12: Authorized localized corner watermark blur vs fail-closed preservation on mandatory/unlicensed mark
  - A9: Post-render decoded-frame source-crop correlation (real passes; wrong/tiny/blank/text-only fail closed!)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from motionlab.assets.pdf_regions import build_product_proof_by_feature, extract_verified_region_crop
from motionlab.composition.hook_competition import rank_rendered_hooks_blind
from motionlab.composition.multi_source_timeline import (
    MultiSourceEditMap,
    apply_localized_watermark_blur_if_authorized,
    evaluate_watermark_authorization,
)
from motionlab.composition.scene_graph_v3 import (
    build_scene_graph_v3_from_plan,
    compile_scene_v3_html,
    solve_v3_layout,
)
from motionlab.semantic.creative_director import build_intent_brief
from motionlab.semantic.evidence_planner import (
    build_atomic_claim_graph,
    propose_model_driven_plans,
    retrieve_evidence_for_claims,
)
from motionlab.semantic.evidence_verifier import (
    match_source_crop_in_decoded_frame,
    verify_rendered_claim_coverage,
)
from motionlab.semantic.speech_dedup import align_source_dialogue_to_intent, verify_required_claims
from motionlab.semantic.video_evidence import detect_static_corner_watermarks


def _make_textured_doc_png(path: Path, title: str, seed: int = 1) -> str:
    """Create a crisp high-contrast synthetic document page image (deterministic, non-blank)."""
    im = Image.new("RGB", (900, 1200), (245, 242, 234))
    dr = ImageDraw.Draw(im)
    dr.rectangle([40, 40, 860, 180], fill=(22, 56, 46))
    dr.text((70, 90), title, fill=(250, 248, 244))
    rng = np.random.default_rng(seed)
    for k in range(10):
        y = 240 + k * 85
        w_bar = int(rng.integers(220, 740))
        col = (181, 143, 82) if (k + seed) % 3 == 0 else (22, 56, 46)
        dr.rectangle([60, y, 60 + w_bar, y + 36], fill=col)
        dr.rectangle([60, y + 46, 820, y + 52], fill=(173, 196, 181))
    im.save(path, "PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mp4_from_frames(path: Path, frames_bgr: list[np.ndarray], fps: float = 15.0) -> None:
    h, w = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames_bgr:
        vw.write(fr)
    vw.release()


class TestR5R6EvidenceCreativeIntelligence(unittest.TestCase):
    def test_a1_source_alignment_completeness_and_critical_claim_drop_mutation(self) -> None:
        """A1: Separates source completeness from edited recall; fails closed when negation/date dropped."""
        intended = (
            "Tell me I'm not the only one checking before groceries. "
            "Rent is due Tuesday and I don't get paid until Friday. "
            "That's why I created BrandX. "
            "I left the link in the caption."
        )
        words_by_src = {
            "A": [
                {"word_id": 0, "word": "Tell", "source_start_s": 0.1, "source_end_s": 0.3},
                {"word_id": 1, "word": "me", "source_start_s": 0.3, "source_end_s": 0.5},
                {"word_id": 2, "word": "I'm", "source_start_s": 0.5, "source_end_s": 0.7},
                {"word_id": 3, "word": "not", "source_start_s": 0.7, "source_end_s": 0.9},
                {"word_id": 4, "word": "the", "source_start_s": 0.9, "source_end_s": 1.0},
                {"word_id": 5, "word": "only", "source_start_s": 1.0, "source_end_s": 1.2},
                {"word_id": 6, "word": "one", "source_start_s": 1.2, "source_end_s": 1.4},
                {"word_id": 7, "word": "checking", "source_start_s": 1.4, "source_end_s": 1.7},
                {"word_id": 8, "word": "before", "source_start_s": 1.7, "source_end_s": 2.0},
                {"word_id": 9, "word": "groceries.", "source_start_s": 2.0, "source_end_s": 2.5},
                {"word_id": 10, "word": "Rent", "source_start_s": 2.7, "source_end_s": 3.0},
                {"word_id": 11, "word": "is", "source_start_s": 3.0, "source_end_s": 3.1},
                {"word_id": 12, "word": "due", "source_start_s": 3.1, "source_end_s": 3.3},
                {"word_id": 13, "word": "Tuesday", "source_start_s": 3.3, "source_end_s": 3.8},
                {"word_id": 14, "word": "and", "source_start_s": 3.8, "source_end_s": 4.0},
                {"word_id": 15, "word": "I", "source_start_s": 4.0, "source_end_s": 4.1},
                {"word_id": 16, "word": "don't", "source_start_s": 4.1, "source_end_s": 4.4},
                {"word_id": 17, "word": "get", "source_start_s": 4.4, "source_end_s": 4.6},
                {"word_id": 18, "word": "paid", "source_start_s": 4.6, "source_end_s": 4.9},
                {"word_id": 19, "word": "until", "source_start_s": 4.9, "source_end_s": 5.2},
                {"word_id": 20, "word": "Friday.", "source_start_s": 5.2, "source_end_s": 5.7},
            ],
            "B": [
                {"word_id": 0, "word": "I", "source_start_s": 0.2, "source_end_s": 0.3},
                {"word_id": 1, "word": "left", "source_start_s": 0.3, "source_end_s": 0.6},
                {"word_id": 2, "word": "the", "source_start_s": 0.6, "source_end_s": 0.7},
                {"word_id": 3, "word": "link", "source_start_s": 0.7, "source_end_s": 1.0},
                {"word_id": 4, "word": "in", "source_start_s": 1.0, "source_end_s": 1.1},
                {"word_id": 5, "word": "the", "source_start_s": 1.1, "source_end_s": 1.2},
                {"word_id": 6, "word": "caption.", "source_start_s": 1.2, "source_end_s": 1.8},
            ],
        }
        al = align_source_dialogue_to_intent(words_by_src, intended)
        self.assertEqual(al["intended_sentences_total"], 4)
        self.assertEqual(al["available_sentences_count"], 3)
        self.assertEqual(al["missing_or_uncertain_sentences_count"], 1)
        self.assertEqual(al["source_completeness_status"], "MISSING_SOURCE_DIALOGUE")

        # Intact available words -> retained ok, release blocked by missing sentence
        edited_ok = words_by_src["A"] + words_by_src["B"]
        chk_ok = verify_required_claims(al, edited_ok, product_name="BrandX")
        self.assertTrue(chk_ok["retained_critical_speech_ok"])
        self.assertTrue(chk_ok["release_blocked_by_script_completeness"])
        self.assertEqual(chk_ok["status"], "RELEASE_BLOCKED_MISSING_SOURCE_DIALOGUE")

        # Mutation: drop "not" or "Tuesday" -> MUST FAIL CLOSED!
        edited_mut = [w for w in edited_ok if w["word"].lower() not in ("not", "tuesday")]
        chk_mut = verify_required_claims(al, edited_mut, product_name="BrandX")
        self.assertFalse(chk_mut["retained_critical_speech_ok"])
        self.assertIn("not", chk_mut["dropped_critical_tokens"])
        self.assertIn("tuesday", chk_mut["dropped_critical_tokens"])
        self.assertEqual(chk_mut["status"], "FAIL_DROPPED_CRITICAL_SPEECH")

    def test_a4_pdf_region_crop_fail_closed_and_feature_proof_separation(self) -> None:
        """A4: Region cropper rejects blank/tampered/OOB/tiny crops; workbook never grants Notion live UI proof."""
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            doc_p = tdir / "page07.png"
            sha = _make_textured_doc_png(doc_p, "Page 07 Proof", seed=7)
            rec = {
                "asset_id": "wb_p07",
                "asset_type": "workbook_page",
                "resolved_path": str(doc_p),
                "sha256": sha,
                "owner_authorized": True,
                "pii_redacted": True,
                "license": "OWNER_AUTHORIZED",
            }
            crop = extract_verified_region_crop(
                asset_record=rec,
                region_id="hero_band",
                norm_bbox=[0.05, 0.04, 0.95, 0.70],
                out_dir=tdir / "crops",
                expected_visible_facts=["Page 07 Proof"],
                page_number=7,
            )
            self.assertTrue(Path(crop["crop_path"]).is_file())
            self.assertGreater(crop["pixel_std"], 10.0)

            # Tampered SHA must raise ValueError
            with self.assertRaises(ValueError):
                extract_verified_region_crop(asset_record={**rec, "sha256": "0" * 64}, region_id="bad_sha", norm_bbox=[0.05, 0.05, 0.95, 0.7], out_dir=tdir / "crops")
            # Tiny illegible bbox must raise ValueError
            with self.assertRaises(ValueError):
                extract_verified_region_crop(asset_record=rec, region_id="tiny", norm_bbox=[0.1, 0.1, 0.18, 0.18], out_dir=tdir / "crops")
            # Blank solid image must raise ValueError
            blank_p = tdir / "blank.png"
            Image.new("RGB", (600, 800), (255, 255, 255)).save(blank_p, "PNG")
            blank_rec = {**rec, "asset_id": "blank", "resolved_path": str(blank_p), "sha256": hashlib.sha256(blank_p.read_bytes()).hexdigest()}
            with self.assertRaises(ValueError):
                extract_verified_region_crop(asset_record=blank_rec, region_id="blank_reg", norm_bbox=[0.1, 0.1, 0.9, 0.9], out_dir=tdir / "crops")

            # Feature-level proof ledger: workbook pages verify workbook ONLY, interactive UI stays PRODUCT_UI_PROOF_BLOCKED!
            ledger = build_product_proof_by_feature(product_name="Demo System", verified_assets=[rec])
            self.assertTrue(ledger["workbook_verified"])
            self.assertFalse(ledger["live_action_ui_verified"])
            self.assertIn("PRODUCT_UI_PROOF_BLOCKED", ledger["release_blockers"])

    def test_a2_a3_atomic_splitting_relevance_rejection_and_causal_plan_switching(self) -> None:
        """D01/D03/D11: Generic cooking clause splitting, unrelated asset rejection, and causal plan switching."""
        # 1. Generic non-finance cooking multi-step sentence splits cleanly into 3 atomic claims!
        cook_words = [
            {"word_id": i, "word": w, "output_start_s": 0.5 + i * 0.28, "output_end_s": 0.76 + i * 0.28}
            for i, w in enumerate(["First,", "pat", "the", "steak", "completely", "dry,", "sear", "it", "in", "hot", "skillet,", "and", "rest", "for", "five", "minutes."])
        ]
        brief_cook = build_intent_brief(claims=[], job_config={"niche_hint": "cooking", "product_name": None}, product_asset_status={})
        cg_cook = build_atomic_claim_graph(edited_words=cook_words, intent_brief=brief_cook)
        self.assertGreaterEqual(len(cg_cook["claims"]), 3)

        # 2. Irrelevant asset rejected cleanly (zero blind manifest-order assignment!)
        saas_tokens = ["Rendering", "slowly", "again?", "This", "tool", "shows", "you", "how", "to", "export", "video", "bitrate", "cleanly."]
        saas_words = [
            {"word_id": i, "word": w, "output_start_s": 0.2 + i * 0.3, "output_end_s": 0.45 + i * 0.3}
            for i, w in enumerate(saas_tokens)
        ]
        brief_saas = build_intent_brief(claims=[], job_config={"niche_hint": "product_saas", "product_name": "ClipTool"}, product_asset_status={"has_verified_real_product_assets": True})
        cg_saas = build_atomic_claim_graph(edited_words=saas_words, intent_brief=brief_saas)
        irrelevant_asset = {
            "asset_id": "cat_photo",
            "asset_type": "workbook_page",
            "verified": True,
            "authorized": True,
            "supported_features": ["feline", "pet_care", "grooming"],
        }
        ev_irr = retrieve_evidence_for_claims(claim_graph=cg_saas, verified_assets=[irrelevant_asset])
        self.assertTrue(all(r["status"] != "MATCHED_VERIFIED_ASSET" for r in ev_irr["results"]))

        # 3. Adding a relevant verified asset causally switches selected plan to zoomed_region_crops!
        relevant_asset = {
            "asset_id": "export_panel_ui",
            "asset_type": "dashboard_screenshot",
            "verified": True,
            "authorized": True,
            "supported_features": ["export", "video", "bitrate", "toolbar"],
        }
        ev_rel = retrieve_evidence_for_claims(claim_graph=cg_saas, verified_assets=[relevant_asset])
        plan_no_asset = propose_model_driven_plans(intent_brief=brief_saas, claim_graph=cg_saas, evidence_results=ev_irr, model_router=None)
        plan_with_asset = propose_model_driven_plans(intent_brief=brief_saas, claim_graph=cg_saas, evidence_results=ev_rel, model_router=None)
        self.assertNotEqual(plan_no_asset["selected_plan"]["product_presentation"], "zoomed_region_crops")
        self.assertEqual(plan_with_asset["selected_plan"]["product_presentation"], "zoomed_region_crops")

    def test_a7_blind_hook_competition_permutation_invariance_and_pixel_swap_causality(self) -> None:
        """D02: Blind hook ranker is permutation-invariant and follows actual rendered pixels, NOT candidate_id/mode!"""
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            w, h = 270, 480
            # Base footage: clean neutral gray gradient
            base_frames = [np.full((h, w, 3), (90, 95, 100), dtype=np.uint8) for _ in range(45)]
            base_mp4 = tdir / "base.mp4"
            _write_mp4_from_frames(base_mp4, base_frames)

            # Hook GOOD: crisp animated graphic card in lower safe zone with high contrast & temporal evolution
            good_frames = []
            for f in range(45):
                fr = base_frames[f].copy()
                cv2.rectangle(fr, (18, 280), (242, 390), (245, 242, 234), -1)
                cv2.rectangle(fr, (24, 295), (24 + min(200, 40 + f * 4), 325), (46, 56, 22), -1)
                cv2.putText(fr, "TUE -> FRI", (32, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 56, 46), 3)
                good_frames.append(fr)
            good_mp4 = tdir / "good.mp4"
            _write_mp4_from_frames(good_mp4, good_frames)

            # Hook WEAK: nearly unchanged dull frame (boring static)
            weak_frames = [np.full((h, w, 3), (92, 96, 101), dtype=np.uint8) for _ in range(45)]
            weak_mp4 = tdir / "weak.mp4"
            _write_mp4_from_frames(weak_mp4, weak_frames)

            # Hook BLACK: defective black frames -> rejected by feasibility gate!
            black_frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(45)]
            black_mp4 = tdir / "black.mp4"
            _write_mp4_from_frames(black_mp4, black_frames)

            cands_order1 = [
                {"candidate_id": "hook_A", "treatment_mode": "mode_x", "preview_mp4": str(weak_mp4), "alpha_verified": True, "face_clear": True},
                {"candidate_id": "hook_B", "treatment_mode": "mode_y", "preview_mp4": str(good_mp4), "alpha_verified": True, "face_clear": True},
                {"candidate_id": "hook_C", "treatment_mode": "mode_z", "preview_mp4": str(black_mp4), "alpha_verified": True, "face_clear": True},
            ]
            # Shuffled & renamed IDs -> MUST still pick exact same winning pixels (good_mp4)!
            cands_order2 = [
                {"candidate_id": "renamed_Z", "treatment_mode": "mode_a", "preview_mp4": str(black_mp4), "alpha_verified": True, "face_clear": True},
                {"candidate_id": "renamed_X", "treatment_mode": "mode_b", "preview_mp4": str(good_mp4), "alpha_verified": True, "face_clear": True},
                {"candidate_id": "renamed_Y", "treatment_mode": "mode_c", "preview_mp4": str(weak_mp4), "alpha_verified": True, "face_clear": True},
            ]
            r1 = rank_rendered_hooks_blind(rendered_hooks=cands_order1, base_mezzanine_mp4=base_mp4, face_bounds=None, work_dir=tdir / "w1", model_router=None, seed=42)
            r2 = rank_rendered_hooks_blind(rendered_hooks=cands_order2, base_mezzanine_mp4=base_mp4, face_bounds=None, work_dir=tdir / "w2", model_router=None, seed=42)
            self.assertEqual(r1["winner_candidate_id"], "hook_B")
            self.assertEqual(r2["winner_candidate_id"], "renamed_X")
            self.assertEqual(r1["winner_preview_sha256"], r2["winner_preview_sha256"])

            # Pixel swap causality: give good_mp4 to hook_A and weak_mp4 to hook_B -> winner MUST flip to hook_A!
            cands_swapped = [
                {"candidate_id": "hook_A", "treatment_mode": "mode_x", "preview_mp4": str(good_mp4), "alpha_verified": True, "face_clear": True},
                {"candidate_id": "hook_B", "treatment_mode": "mode_y", "preview_mp4": str(weak_mp4), "alpha_verified": True, "face_clear": True},
            ]
            r3 = rank_rendered_hooks_blind(rendered_hooks=cands_swapped, base_mezzanine_mp4=base_mp4, face_bounds=None, work_dir=tdir / "w3", model_router=None, seed=42)
            self.assertEqual(r3["winner_candidate_id"], "hook_A")

    def test_a8_watermark_localized_blur_blocked_vs_authorized_non_mandatory(self) -> None:
        """D12: User edit preference alone NEVER authorizes blur; verified non-mandatory license blurs ONLY corner ROI."""
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            p_src = tdir / "marked.mp4"
            w, h = 270, 480
            frames = []
            for f in range(20):
                fr = np.full((h, w, 3), (75, 80, 86), dtype=np.uint8)
                cv2.rectangle(fr, (30, 80), (180, 260), (140, 130, 125), -1)
                cx, cy = int(w * 0.88), int(h * 0.91)
                pts = np.array([[cx, cy - 12], [cx + 4, cy - 4], [cx + 12, cy], [cx + 4, cy + 4], [cx, cy + 12], [cx - 4, cy + 4], [cx - 12, cy], [cx - 4, cy - 4]])
                cv2.fillPoly(fr, [pts], (245, 245, 245))
                frames.append(fr)
            _write_mp4_from_frames(p_src, frames)
            roi = detect_static_corner_watermarks(p_src, num_samples=10)
            self.assertTrue(roi["detected_bright_static_mark"])

            # Case 1: User asks for localized_blur, but no license / mandatory mark -> FAIL CLOSED!
            auth_blocked = evaluate_watermark_authorization(
                {"A": roi},
                authorization_doc={"clean_export_authorized": False, "mandatory_provenance_mark": True, "edit_preference_method": "localized_blur"},
            )
            self.assertEqual(auth_blocked["status"], "BLOCKED_WATERMARK_AUTHORIZATION")
            self.assertFalse(auth_blocked["localized_blur_authorized"])
            res_blocked = apply_localized_watermark_blur_if_authorized(
                video_path=p_src, out_path=tdir / "blocked_out.mp4", watermark_auth=auth_blocked, source_roi=roi,
            )
            self.assertFalse(res_blocked["applied"])

            # Case 2: Explicitly authorized non-mandatory corner mark -> tight ROI blur, outside ROI preserved!
            auth_ok = evaluate_watermark_authorization(
                {"A": roi},
                authorization_doc={"clean_export_authorized": True, "mandatory_provenance_mark": False, "preferred_method": "localized_blur", "license_basis": "OWNER_STUDIO_ASSET_LICENSE"},
            )
            self.assertEqual(auth_ok["status"], "AUTHORIZED_CLEANUP_ALLOWED")
            self.assertTrue(auth_ok["localized_blur_authorized"])
            res_ok = apply_localized_watermark_blur_if_authorized(
                video_path=p_src, out_path=tdir / "blurred_out.mp4", watermark_auth=auth_ok, source_roi=roi,
            )
            self.assertTrue(res_ok["applied"])
            self.assertTrue(res_ok["outside_roi_preserved"])
            self.assertGreater(res_ok["inside_roi_change_mad"], 0.5)

    def test_a9_observed_crop_correlation_real_vs_wrong_and_tiny_negative(self) -> None:
        """A9/D05: Decoded-frame crop correlation verifies real legible crop and rejects wrong/tiny crops & text-only payoff."""
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            crop_real = tdir / "real_crop.png"
            crop_wrong = tdir / "wrong_crop.png"
            _make_textured_doc_png(crop_real, "Real Page 10", seed=10)
            _make_textured_doc_png(crop_wrong, "Wrong Page 99", seed=99)

            # Composite real_crop cleanly at 76% width inside a 1080x1920 studio frame
            stage = np.full((1920, 1080, 3), (46, 56, 22), dtype=np.uint8)
            img_real = cv2.imread(str(crop_real))
            tw = int(1080 * 0.76)
            th = int(tw * img_real.shape[0] / img_real.shape[1])
            resized = cv2.resize(img_real, (tw, th), interpolation=cv2.INTER_AREA)
            y0, x0 = 380, (1080 - tw) // 2
            stage[y0 : y0 + th, x0 : x0 + tw] = resized

            hit_real = match_source_crop_in_decoded_frame(stage, crop_real)
            hit_wrong = match_source_crop_in_decoded_frame(stage, crop_wrong)
            self.assertTrue(hit_real["matched"], f"Real legible crop should match: {hit_real}")
            self.assertFalse(hit_wrong["matched"], f"Wrong page must fail correlation: {hit_wrong}")

            # Tiny unreadable 20%-wide thumbnail must fail scale check!
            stage_tiny = np.full((1920, 1080, 3), (46, 56, 22), dtype=np.uint8)
            tw_s = int(1080 * 0.22)
            th_s = int(tw_s * img_real.shape[0] / img_real.shape[1])
            stage_tiny[400 : 400 + th_s, 400 : 400 + tw_s] = cv2.resize(img_real, (tw_s, th_s))
            hit_tiny = match_source_crop_in_decoded_frame(stage_tiny, crop_real)
            self.assertFalse(hit_tiny["matched"], f"Tiny 22% thumbnail must fail mobile legibility: {hit_tiny}")

            # Text-only emotional payoff MUST emit TEXT_ONLY_ON_MATERIAL_CLAIM defect!
            mp4_p = tdir / "payoff_text.mp4"
            _write_mp4_from_frames(mp4_p, [stage] * 30, fps=15.0)
            cg = {
                "total_material_weight": 1.0,
                "claims": [{
                    "claim_id": "c_pay",
                    "role": "emotional_resolution_payoff",
                    "materiality": "supporting",
                    "weight": 1.0,
                    "is_material": True,
                    "required_evidence_kind": "ILLUSTRATIVE_EXAMPLE_OR_CONTEXT",
                    "start_s": 0.1,
                    "end_s": 1.8,
                }],
            }
            sg = {"scenes": [{"scene_id": "sc_pay", "claim_ids": ["c_pay"], "treatment_mode": "phone_action_callout", "start_s": 0.1, "end_s": 1.8}]}
            obs = verify_rendered_claim_coverage(final_mp4=mp4_p, base_mezzanine_mp4=None, claim_graph=cg, scene_graph=sg, ev_dir=tdir / "ev", model_router=None, run_vlm_on_claims=False)
            self.assertEqual(obs["rows"][0]["observed_evidence_class"], "TEXT_ONLY")
            self.assertTrue(any(d["category"] == "TEXT_ONLY_ON_MATERIAL_CLAIM" for d in obs["defects"]))

    def test_11a_universal_14_niche_transfer_zero_finance_leak_and_16x9_layout(self) -> None:
        """11A: Cold-catalog transfer across 14 distinct niches + 16:9 landscape emits ZERO leaked Olivia/finance text."""
        niches_and_lines = [
            ("product_saas", "Filter tasks by priority, click Assign Sprint, and watch capacity update."),
            ("history", "The Estates General convened in May before July storming of the Bastille."),
            ("science", "Hydraulic fluid transfers pedal force directly to the brake caliper piston."),
            ("education", "Subtract six from both sides, then divide by two to isolate x."),
            ("screen_tutorial", "Open export settings, choose H264 codec, and confirm bitrate."),
            ("gameplay", "Watch the stamina gauge drop before timing the shield parry."),
            ("cooking", "Pat the fillet dry, sear ninety seconds, and rest on a warm rack."),
            ("travel", "Board the morning express from Kyoto and arrive in Kanazawa before noon."),
            ("comedy", "He dropped the umbrella once and yelled not again at the crowd."),
            ("storytelling", "She paused by the harbor window as the fog rolled across the pier."),
            ("sports", "The guard sets a high screen while the wing rotates to the corner."),
            ("silent_art", ""),
            ("documentary", "Chapter two examines the 2008 industrial shift across three cities."),
            ("finance", "Compare upcoming commitments against your remaining balance carefully."),
        ]
        for niche, line in niches_and_lines:
            words = [
                {"word_id": i, "word": w, "output_start_s": 0.2 + i * 0.25, "output_end_s": 0.42 + i * 0.25}
                for i, w in enumerate(line.split())
            ]
            brief = build_intent_brief(claims=[], job_config={"niche_hint": niche, "product_name": None}, product_asset_status={})
            cg = build_atomic_claim_graph(edited_words=words, intent_brief=brief)
            ev = retrieve_evidence_for_claims(claim_graph=cg, verified_assets=[])
            ps = propose_model_driven_plans(intent_brief=brief, claim_graph=cg, evidence_results=ev, model_router=None)
            is_land = niche in ("screen_tutorial", "documentary", "gameplay")
            W, H = (1920, 1080) if is_land else (1080, 1920)
            emap = MultiSourceEditMap(
                schema_version="1.0.0", sources=[], target_width=W, target_height=H,
                fps_num=30, fps_den=1, sample_rate=48000, channels=2,
                output_total_frames=240, output_duration_s=8.0, output_total_samples=384000,
                kept_spans=[], cuts=[], map_hash=f"map_{niche}",
            )
            sg = build_scene_graph_v3_from_plan(
                selected_plan=ps["selected_plan"], claim_graph=cg, evidence_results=ev,
                cropped_regions_by_claim={}, winning_hook_scene=None, intent_brief=brief,
                edit_map=emap, face_bounds=None, bridge_span=None,
            )
            self.assertGreaterEqual(len(sg["scenes"]), 1, f"Niche {niche} produced empty scene graph")
            for sc in sg["scenes"]:
                html_s, transp = compile_scene_v3_html(sc, width=W, height=H, fps=30)
                self.assertTrue(transp)
                if niche != "finance":
                    low = html_s.lower()
                    for forbidden in ("olivia", "next paycheck plan", "rent due", "payday"):
                        self.assertNotIn(forbidden, low, f"Leaked finance/Olivia token {forbidden!r} in niche {niche}!")
                if niche == "silent_art":
                    self.assertEqual(sc["treatment_mode"], "clean_speaker_anchor")
            if is_land:
                lay169 = solve_v3_layout(width=1920, height=1080, treatment_mode="causal_sequence_diagram")
                self.assertEqual(lay169["aspect"], "16:9")
                self.assertLess(lay169["y_px"], 1080)


if __name__ == "__main__":
    unittest.main()

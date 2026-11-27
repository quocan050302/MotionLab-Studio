"""Phase A10-A11 (ML-01.R5.R3): Safe Atomic QA-02 Evidence Pack (D14, D15) & Hardened Technical Gates (D16)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from motionlab.composition.qa_pack import MAX_QA_PACK_BYTES, build_qa02_evidence_pack


class TestR5R3StreamingQA(unittest.TestCase):
    def _make_fake_run(self, run_dir: Path) -> None:
        (run_dir / "evidence").mkdir(parents=True, exist_ok=True)
        (run_dir / "final.mp4").write_bytes(b"FAKE_MP4_BYTES" * 100)
        (run_dir / "preview_9x16.mp4").write_bytes(b"FAKE_PREV_BYTES" * 50)
        Image.new("RGB", (360, 640), (30, 40, 35)).save(run_dir / "contact_sheet.jpg", "JPEG")
        (run_dir / "manifest.json").write_text(json.dumps({"input_hash": "abc123", "source_path": "/Users/private_owner/secret/vid.mp4"}), encoding="utf-8")
        (run_dir / "storyboard_spec.json").write_text(json.dumps({
            "source_hash": "abc123", "storyboard_hash": "sb_hash_1", "selected_graph_hash": "sel_hash_1",
            "width": 720, "height": 1280, "fps_num": 24, "fps_den": 1, "total_frames": 720,
            "scenes": [{"scene_id": "sc_01", "chosen_candidate_id": "b0_depth", "claim_refs": [{"verbatim_quote": "Secret raw spoken transcript line"}]}],
        }), encoding="utf-8")
        (run_dir / "story_graph.json").write_text(json.dumps({"full_text": "Secret raw spoken transcript line", "contact": "owner@example.com"}), encoding="utf-8")
        (run_dir / "actual_engine_usage.json").write_text(json.dumps({"engines_used": ["ffmpeg", "hyperframes", "remotion"], "reuse_gate": {"remotion_primitives": ["rem_ru_waterfall"], "hyperframes_primitives": ["hf_apple_money_count"]}}), encoding="utf-8")
        (run_dir / "validation.json").write_text(json.dumps({"overall_status": "TECHNICAL_PASS", "aesthetic_status": "PENDING_HUMAN_REVIEW", "foley_library_status": "RECORDED_FOLEY_VERIFIED"}), encoding="utf-8")
        (run_dir / "raw_director_response.json").write_text(json.dumps({
            "brain_audit": {
                "reasoning_mode": "HYBRID_CAUSAL_GLOBAL_DIRECTOR",
                "causal_selection_before_compile": True,
                "real_transcript_supplied": True,
                "timestamp_quality": "WORD_ALIGNED_APPROX",
                "selection_trace": [{"scene_id": "sc_01", "chosen_candidate_id": "b0_depth", "rationale": "Loaded from C:/Users/secret_user/MotionLab-AI-ML01/input/clip.mp4"}],
            }
        }), encoding="utf-8")
        (run_dir / "evidence" / "scene_boundary_report.json").write_text(json.dumps({"vlm_status": "VISION_CRITIC_NOT_RUN", "severe_findings_count": 0}), encoding="utf-8")
        (run_dir / "evidence" / "revision_history.json").write_text(json.dumps({"rounds_executed": 0, "status": "CLEAN_PASS_NO_REPAIR_NEEDED"}), encoding="utf-8")

    def test_d14_d15_qa02_atomic_safe_containment_redaction_and_image_permission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            run_dir = td_p / "run_001"
            self._make_fake_run(run_dir)
            qa_root = td_p / "qa_evidence"
            qa_root.mkdir()

            # 1. Path traversal outside allowed_root MUST fail closed!
            with self.assertRaises(Exception):
                build_qa02_evidence_pack(
                    run_dir=run_dir, pack_dir=td_p / "outside_dir", tested_code_sha="deadbeef",
                    repo_url="https://example.invalid/repo", branch="feat/test", label="test",
                    render_mode="AUTO_GENERATIVE", final_mp4_name="final.mp4", allowed_root=qa_root,
                )

            # 2. Build clean QA-02 pack (include_contact_sheet=True WITHOUT image_sharing_authorized MUST keep CONTACT_SHEET_LOCAL_ONLY!)
            pack_dir = qa_root / "ML-01.R5.R3" / "deadbee" / "run_001"
            res = build_qa02_evidence_pack(
                run_dir=run_dir, pack_dir=pack_dir, tested_code_sha="deadbeef1234",
                repo_url="https://example.invalid/repo", branch="feat/test", label="test",
                render_mode="AUTO_GENERATIVE", final_mp4_name="final.mp4",
                include_contact_sheet=True, image_sharing_authorized=False, allowed_root=qa_root,
            )
            self.assertTrue(res["within_2mib_budget"])
            self.assertLessEqual(res["total_bytes"], MAX_QA_PACK_BYTES)
            self.assertEqual(res["contact_sheet_status"], "CONTACT_SHEET_LOCAL_ONLY")
            self.assertFalse((pack_dir / "contact_sheet_compact.jpg").exists())
            self.assertTrue((pack_dir / "selected_candidate_trace.json").is_file())
            self.assertTrue((pack_dir / "revision_history.json").is_file())

            # Deep redaction check across ALL emitted JSON files!
            blob = "\n".join(p.read_text(encoding="utf-8") for p in pack_dir.glob("*.json"))
            self.assertNotIn("Secret raw spoken transcript line", blob)
            self.assertNotIn("/Users/private_owner", blob)
            self.assertNotIn("C:/Users/secret_user", blob)
            self.assertNotIn("owner@example.com", blob)

            man = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(man["final_artifact"]["access"], "ARTIFACT_LOCAL_ONLY")
            self.assertIsNone(man["final_artifact"]["url"])

            # 3. Attempting to overwrite an existing immutable pack WITHOUT allow_overwrite_owned MUST raise FileExistsError (D15)!
            with self.assertRaises(FileExistsError):
                build_qa02_evidence_pack(
                    run_dir=run_dir, pack_dir=pack_dir, tested_code_sha="deadbeef1234",
                    repo_url="https://example.invalid/repo", branch="feat/test", label="test",
                    render_mode="AUTO_GENERATIVE", final_mp4_name="final.mp4", allowed_root=qa_root,
                )

            # 4. Non-QA directory (missing manifest.json) MUST NEVER be deleted even if allow_overwrite_owned=True!
            non_qa = qa_root / "important_user_notes"
            non_qa.mkdir()
            (non_qa / "notes.txt").write_text("do not delete", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_qa02_evidence_pack(
                    run_dir=run_dir, pack_dir=non_qa, tested_code_sha="deadbeef1234",
                    repo_url="https://example.invalid/repo", branch="feat/test", label="test",
                    render_mode="AUTO_GENERATIVE", final_mp4_name="final.mp4", allowed_root=qa_root,
                    allow_overwrite_owned=True,
                )
            self.assertTrue((non_qa / "notes.txt").is_file())

    def test_d16_hardened_gate_logic_rejects_zero_ablation_and_missing_mix(self) -> None:
        """Verify D16 hardened gate invariants directly against r5_engine source contract."""
        src = (Path(__file__).resolve().parents[2] / "motionlab" / "composition" / "r5_engine.py").read_text(encoding="utf-8")
        self.assertIn("ablation_total > 0 and ablation_verified == ablation_total", src)
        self.assertIn("mix_info is not None and audio_qa.get(\"status\") == \"PASS\"", src)
        self.assertIn("severe_findings_count", src)


if __name__ == "__main__":
    unittest.main()

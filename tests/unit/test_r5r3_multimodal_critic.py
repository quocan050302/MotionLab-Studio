"""Phase A9 (ML-01.R5.R3): Streaming Pixel Critic (D10), All-Boundary/Internal Flash Detection (D11) & VLM Adapter (D09)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from motionlab.composition.generative_schema import GENERATIVE_SCHEMA_VERSION, SceneSpec, StoryboardSpec
from motionlab.composition.visual_critic import (
    inspect_local_vlm,
    run_local_vlm_critic,
    run_objective_rendered_critic,
)


def _write_test_mp4(path: Path, n_frames: int, w: int = 180, h: int = 320, flash_frames: set[int] | None = None, post_spike_frame: int | None = None) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    flashes = flash_frames or set()
    for i in range(n_frames):
        base = 55 + int(10 * np.sin(i * 0.08))
        fr = np.full((h, w, 3), base, dtype=np.uint8)
        if i in flashes:
            fr[:, :] = 250  # Harsh white flash!
        if post_spike_frame is not None and i >= post_spike_frame:
            fr[:, :] = 225  # Sudden jump right after boundary!
        vw.write(fr)
    vw.release()
    return path


class TestR5R3MultimodalCritic(unittest.TestCase):
    def test_d09_vlm_adapter_probes_manifest_honestly_and_never_fakes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mdir = Path(td)
            # Missing manifest -> VISION_CRITIC_NOT_RUN
            v0 = inspect_local_vlm(models_dir=mdir)
            self.assertFalse(v0["available"])
            self.assertEqual(v0["status"], "VISION_CRITIC_NOT_RUN")

            # Text-only manifest -> VISION_CRITIC_NOT_RUN (never treats text LLM as VLM!)
            (mdir / "manifest.json").write_text(json.dumps({"models": {"qwen_3b": {"filename": "qwen.gguf"}}}), encoding="utf-8")
            v1 = inspect_local_vlm(models_dir=mdir)
            self.assertFalse(v1["available"])
            self.assertEqual(v1["status"], "VISION_CRITIC_NOT_RUN")

            # run_local_vlm_critic must report VISION_CRITIC_NOT_RUN without simulating findings
            res = run_local_vlm_critic(frame_samples=[], narrative_summary="test", models_dir=mdir)
            self.assertEqual(res["status"], "VISION_CRITIC_NOT_RUN")
            self.assertEqual(res["frames_fed"], 0)

    def test_d10_d11_streaming_critic_catches_internal_flash_and_post_boundary_spike(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src = _write_test_mp4(td_p / "src.mp4", n_frames=72)
            # Broken final: internal white flash at f18 + sudden unmotivated jump 2 frames AFTER boundary f48 (at f50!)
            fin_bad = _write_test_mp4(td_p / "fin_bad.mp4", n_frames=72, flash_frames={18}, post_spike_frame=50)
            fin_ok = _write_test_mp4(td_p / "fin_ok.mp4", n_frames=72)

            sb = StoryboardSpec(
                GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", "src", "tr", "sg", "finance", 0.9, "v",
                180, 320, 24, 1, 72, 42, {},
                [
                    SceneSpec("sc_01", 0, 48, "hook_contrast", "live_clean_bridge", "s1"),
                    SceneSpec("sc_02", 48, 72, "allocation_step", "live_clean_bridge", "s2"),
                ],
            )
            cont = {
                "boundaries": [{
                    "boundary_id": "bnd_00_f048",
                    "boundary_frame": 48,
                    "from_scene_id": "sc_01",
                    "to_scene_id": "sc_02",
                    "from_stage_mode": "live_clean_bridge",
                    "to_stage_mode": "live_clean_bridge",
                    "boundary_class": "MATCH_ACTION",
                    "bridge_direction": "shared_morph",
                    "handle_frames": 0,
                }]
            }

            rep_bad = run_objective_rendered_critic(
                source_mp4=src, final_mp4=fin_bad, storyboard=sb, continuous_timeline=cont,
                roi_tracks=None, out_dir=td_p / "ev_bad", ring_buffer_size=12,
            )
            self.assertTrue(rep_bad["streaming_mode"])
            self.assertEqual(rep_bad["peak_ring_buffer_frames"], 12)
            self.assertGreaterEqual(rep_bad["internal_flash_count"], 1)
            self.assertGreaterEqual(rep_bad["boundary_window_spikes"], 1)
            self.assertGreaterEqual(rep_bad["severe_findings_count"], 1)
            self.assertEqual(rep_bad["objective_pixel_critic_status"], "FINDINGS_FLAGGED")

            rep_ok = run_objective_rendered_critic(
                source_mp4=src, final_mp4=fin_ok, storyboard=sb, continuous_timeline=cont,
                roi_tracks=None, out_dir=td_p / "ev_ok", ring_buffer_size=12,
            )
            self.assertEqual(rep_ok["internal_flash_count"], 0)
            self.assertEqual(rep_ok["severe_findings_count"], 0)
            self.assertEqual(rep_ok["objective_pixel_critic_status"], "PASS")

    def test_d10_streaming_critic_handles_long_60s_1440f_clip_bounded_ring(self) -> None:
        """Prove O(window) streaming architecture handles 60s (1,440 frames) without storing full-video BGR lists."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src = _write_test_mp4(td_p / "long_src.mp4", n_frames=1440, w=90, h=160)
            sb = StoryboardSpec(
                GENERATIVE_SCHEMA_VERSION, "AUTO_GENERATIVE", "src", "tr", "sg", "finance", 0.9, "v",
                90, 160, 24, 1, 1440, 42, {}, [SceneSpec("sc_01", 0, 1440, "hook_contrast", "live_clean_bridge", "long")],
            )
            rep = run_objective_rendered_critic(
                source_mp4=src, final_mp4=src, storyboard=sb, continuous_timeline={"boundaries": []},
                roi_tracks=None, out_dir=td_p / "ev_long", ring_buffer_size=16,
            )
            self.assertTrue(rep["streaming_mode"])
            self.assertEqual(rep["frames_inspected"], 1440)
            self.assertEqual(rep["peak_ring_buffer_frames"], 16)
            self.assertEqual(rep["severe_findings_count"], 0)


if __name__ == "__main__":
    unittest.main()

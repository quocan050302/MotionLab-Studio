# ADR — ML-01.R5.R7: Generative Visual Reasoning & Cinematic Composition

- **Date:** 2026-11-28
- **Status:** Implemented (Opt-in `director_strategy: "generative_v4"`)
- **Base:** Rebased cleanly on audited ML-01.R5.R6 (`45040bc06a054ed1adb7116ce7c9f84d9d1ec91d`, tested code `e0ecdecfc7377f22e7ad4184f05eaad0041ff82a`)

## Context & Architectural Decisions

1. **Preserve 100% Legacy & R5.R5/R5.R6 Compatibility via Opt-in Strategy (`generative_v4`):**
   - Existing `--input`, R5.R5 (`v2`), and R5.R6 (`evidence_v3`) pipelines remain untouched and backward-compatible.
   - R5.R7 activates exclusively via `"director_strategy": "generative_v4"` in the job manifest, dispatching cleanly from `motionlab/composition/director_pipeline.py` to `motionlab/composition/director_v4_runner.py`.

2. **Local Model Upgrade with Single-Residency Budget Enforcement (A1 / D08):**
   - Staged verified Apache-2.0 `Qwen3-4B-Instruct-2507` Q4_K_M (2.38 GB) and official `Qwen3-VL-4B-Instruct` Q4_K_M + mmproj Q8_0 (2.81 GB) alongside incumbents (`Qwen2.5-3B`, `Qwen3-VL-2B`) on installed `llama.cpp` b12803.
   - Single-model residency lock (`_SINGLE_MODEL_LOCK`) guarantees text LLM and VLM NEVER run concurrently, maintaining >= 3 GiB RAM reserve on our 31.75 GiB CPU-only host.
   - Model upgrade adoption requires measured raw improvement on the frozen 48-case (16 heldout) + 16-vision-task benchmark (`MODEL_UPGRADE_PROVEN` vs `MODEL_UPGRADE_NOT_PROVEN`).

3. **Global StoryBible & Canonical Sentence Authority (A2 / D03 / D09 / D10 / D12):**
   - Added `motionlab/semantic/story_bible.py` with a single deterministic `canonical_split_sentences()` function (resolving the prior `8/10` vs `9/11` reporting mismatch).
   - Natural chapter segmentation and full-timeline atomic claim decomposition eliminate `claims[:10]` truncation and preserve persistent entity state across chapters (blocking silent cross-chapter rollbacks).

4. **Composable Generative SceneProgram V4 & Dual-Engine Rendering (A3 / A5 / A6 / D04):**
   - Replaced enum-only plan selection with model-authored typed primitive graphs (`generative_visual_planner.py`), finite allowlisted read-only tools, and structural diversity ranking.
   - Compiled validated SceneGraph V4 graphs to **both** Remotion (`SceneGraphV4Composition.tsx`, using MIT `remotion-ui` `Counter` + pure `globalStartFrame + useCurrentFrame()`) and HyperFrames (paused seek-exact GSAP `fromTo` timelines), replacing static icon cards with multi-phase animated action sequences.
   - Emitted authoritative `motion_event_track.json` and exact `[0, N)` `frame_coverage_manifest.json`.

5. **MIT PDF Text BBox Extraction & Verified Worked-Example State Machine (A4 / D11):**
   - Retained zero-AGPL Poppler `pdftoppm`/`pdfinfo` rasterization and added MIT `pdfminer.six` exact text line & bounding-box extraction (`pdf_regions.py::extract_pdf_text_bboxes()`), plus a domain-agnostic deterministic worked-example state machine (`build_verified_example_state_machine()`) that never delegates arithmetic to an LLM.
   - Maintained strict feature-level proof separation (`ACTUAL_PRODUCT_WORKBOOK` vs `PRODUCT_UI_PROOF_BLOCKED` for live Notion UI).

6. **Long-Form Chunked DAG Scheduler & Total Process-Tree Monitoring (A7 / D05 / D09 / D12):**
   - Added `motionlab/composition/longform_scheduler.py` implementing contiguous half-open `[start, end)` chunking, handle trimming, content-addressed cache invalidation, crash-safe verified resume, single final AAC encode, and `TotalProcessTreeMonitor` (capturing Python + FFmpeg + Chromium + llama.cpp child RSS).

7. **Continuous Event-Driven Sonic Score (A8 / D05 / D13):**
   - Extended existing audio stack via `motionlab/audio/continuous_score.py` binding cues directly to `MotionEventTrack` action markers across all chapters (preventing late-chapter cue drop-off and 3-in-a-row motif repetition), adding phase-continuous voice-ducked ambient stems, and validating sample conservation + onset alignment.

8. **Independent Multi-Frame Semantic Comprehension Critic, Sequence Hooks & Dependency-Closure Repair (A10 / D01 / D02 / D06 / D07):**
   - Added `motionlab/semantic/semantic_comprehension_critic.py` removing the global `vlm_calls < 7` cap in favor of per-chapter risk-adaptive budgets, separating render presence / semantic correctness / comprehension proxy / reading exposure / uncertainty (never defaulting `UNKNOWN`/`VLM_ERROR` to PASS), and implementing dependency-closure graph repair with rollback on regression.
   - Extended `hook_competition.py` with `rank_hook_sequences_blind()` evaluating ordered `[OPEN | CHANGE | PAYOFF]` sequences + speech comprehension + honest tie detection.

# Architecture Decision Record — ML-01.R5.R5 (Generative Creative Director & Multimodal Visual Intelligence)

- **Date:** 2026-11-28
- **Status:** Accepted & Implemented
- **Branch:** `feat/ml-01-zero-api-effects` (baseline audited HEAD `d8a915e91001f114fce9498ea253c1f64e1f4b7c`)

---

## ADR-01: Hybrid Single-Residency Local Model Router (`Qwen2.5-3B` Text + `Qwen3-VL-2B` Vision)
- **Context:** R5.R4 lacked any local multimodal VLM (`VISION_CRITIC_NOT_RUN`). Host hardware is CPU-only Windows 11 x64 (`31.75 GiB` RAM total, `4.11–7.42 GiB` available under normal desktop load, `38+ GiB` free disk).
- **Experiment:** Benchmarked `Qwen2.5-3B-Instruct Q4_K_M` vs `Qwen3-VL-2B-Instruct Q4_K_M` on 12 diverse strict-policy reasoning cases (`BM01..BM12`) plus real decoded frame perception and blurred-screen negative refusal (`runs/r5r5_a0_baseline_audit/model_benchmark.json`).
  - `Qwen2.5-3B` text: `12/12` valid JSON, `12/12` raw passed (`100%`, `113.6s`).
  - `Qwen3-VL-2B` text: `12/12` valid JSON (`50.6s`), `1/12` raw passed (`8.3%` — overly permissive on strict policy booleans).
  - `Qwen3-VL-2B + mmproj Q8_0` vision (`llama-mtmd-cli.exe`): `VLM_EXECUTED` in `9.0s/frame` (`2,615 MB` peak RSS), accurately describing real car/phone/person scene (`readable_bank_balance="UNREADABLE"`) and refusing blurred-screen hallucination.
- **Decision:** Adopt `hybrid_qwen2_5_3b_text_plus_qwen3_vl_2b_vision` via `motionlab/semantic/model_router.py`. Keep `Qwen2.5-3B-Instruct Q4_K_M` for structured text creative reasoning and `Qwen3-VL-2B-Instruct Q4_K_M + mmproj Q8_0` for real decoded-frame multimodal perception and independent visual QA critic. Enforce single-model residency (never run text LLM and VLM simultaneously). Defer `Qwen3-VL-4B` (~3 GB + KV) to preserve memory safety for Chromium/FFmpeg rendering.

## ADR-02: Real Production Multi-Source Director CLI (`--job` + `--creative-mode director`)
- **Context:** Baseline defect D01 showed `d8a915e` added `speech_dedup.py`, `multi_source_timeline.py`, and `calendar_editorial.py` without connecting them into production CLI callgraphs.
- **Decision:** Extend `motionlab/cli.py` (`edit` and `plan` subcommands) with opt-in `--job <json>` and `--creative-mode <legacy|director>`, strictly mutually exclusive with legacy `--input`. Route `--job` through `motionlab/composition/director_pipeline.py::run_multisource_director_job`, preserving 100% backward compatibility for single-source `--input --pacing preserve|auto`. Extend `motionlab/pipeline.py::run_validate_existing_run` to validate `multisource_director_v2` runs non-destructively.

## ADR-03: Generalized Arbitrary N-Source Timeline & Strict End-Exclusive Rational Mapping
- **Context:** Baseline defect D04 hardcoded `"A"`/`"B"` source assumptions, a fixed clip B room-tone window `(9.25, 9.75)`, and a `±0.03s` tolerance in `map_source_time_to_output` that risked mapping timestamps inside omitted spans.
- **Decision:** Upgrade `motionlab/composition/multi_source_timeline.py` and `motionlab/semantic/speech_dedup.py` to support arbitrary `N >= 1` source IDs, mixed FPS (`24/1`, `30/1`, `30000/1001`) and sample rates (`44.1/48 kHz`), strict end-exclusive `[start_s, end_s)` mapping, VAD-proven quiet non-speech room-tone selection (fail-closed silent fallback if none verified), and multi-corner local-contrast static watermark detection (`video_evidence.py::detect_static_corner_watermarks`).

## ADR-04: SceneGraph v2 Compositional Operators & Dynamic Layout Solver
- **Context:** Baseline defects D02, D03, D10 revealed repeated dark glass pills at fixed `y=1255`, hardcoded Olivia/finance defaults with silent CTA fallback on unknown kinds, and abrupt cream hard cuts.
- **Decision:** Introduce `motionlab/composition/scene_graph_v2.py` with 6 materially distinct editorial treatment modes (`spatial_editorial_anchor`, `calendar_timeline_gap`, `split_dilemma_metaphor`, `motivated_object_bridge` clip-path sweep, `illustrative_system_board`, `minimal_cta_anchor`) plus `calm_breathing_face`, dynamic W/H + face-ROI + safe-zone layout solving (`solve_dynamic_safe_layout`), fail-closed `ValueError` on unknown modes/kinds (also fixed in `calendar_editorial.py`), and 3 rendered hook proxy candidates (`hook_A`, `hook_B`, `hook_C`) causally bound to the final render.

## ADR-05: Generic `TemporalFactLedger` & Honest Separation of Technical vs Rights/Proof/Human Gates
- **Context:** Baseline defects D05, D08, D11 revealed silent state rollbacks (`$1200 -> $100 -> $1200`), unverified product claims, and blurred pass/blocker boundaries.
- **Decision:** Enforce domain-agnostic `motionlab/semantic/temporal_ledger.py` (blocking unmarked state rollbacks, premature states, and weekday inversions across finance, tutorial, gameplay, and narrative domains while permitting explicit `is_recap=True`). Strictly separate technical PASS from hard delivery blockers (`BLOCKED_WATERMARK_AUTHORIZATION`, `PRODUCT_PROOF_BLOCKED`, `MISSING_SOURCE_DIALOGUE`) and human review (`PENDING_HUMAN_REVIEW`, `LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY`), naming outputs `*_CANDIDATE_9x16.mp4` whenever any hard blocker is present.

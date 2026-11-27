# Architecture Decision Record — MotionLab Studio ML-01.R5.R2

- **Repository:** `https://github.com/quocan050302/MotionLab-Studio`
- **Branch:** `feat/ml-01-zero-api-effects`
- **Base Commit (R5.R1):** `43e68a21d1d5345b88e4d724c4bac8dbcaf033fe`
- **Tested Code Commit (R5.R2):** `04ee4faceb107ee3f2470b52f3734c2826d37efe`
- **Runtime Mode:** 100% Offline / Zero-API (local Faster-Whisper Small, local Qwen2.5-3B-Instruct Q4_K_M via `llama-completion.exe`, local MODNet ONNX, local Remotion 4.0.547 + HyperFrames 0.9.101 + FFmpeg 9.0.2, local CC0-1.0 Foley library).

---

## ADR-01: Stateful Multimodal Narrative Brain & Fail-Closed Candidate Ranker

- **Context:** Raw `qwen2_5_3b_instruct_q4_k_m` output on 30s spoken scripts routinely collapses into repetitive `full_graphic_cutaway`, identical `hf_exploded_stack`, and indiscriminate `cash_flick` cues even on negated or non-monetary beats (empirically observed: 9/10 beats required reconciliation in the canonical cold run).
- **Decision:** Introduce `motionlab/semantic/narrative_brain.py` (`EvidenceIndex`, `normalize_semantic_entities`, `resolve_coreferences`, `NarrativeContract`, `NarrativeStateGraph`, `plan_attention_arc`, versioned `CreativePlaybook` PB-01..PB-07, `propose_scene_graphs`, `rank_scene_candidates`) and pair it with an independent 16-case reasoning evaluation harness (`motionlab/semantic/eval_harness.py`, EV01..EV16).
- **Consequences:** Raw LLM proposals are audited verbatim alongside deterministic reconciled decisions (`raw_director_response.json`). Zero Olivia-specific timestamps, scripts, or amounts exist in runtime; filename-independence and cross-niche generalization remain preserved while hard structural defects (contradicting numbers, unbridged cutaways, happy cash sounds on negations) are rejected fail-closed.

## ADR-02: Single Canonical Numeric State Owner & Integer-Dollar Lock

- **Context:** R5.R1 exhibited D01 (`$1,151.93`), D03 (`-$799` top strip + duplicate bottom ribbon `$402`), and D05 (`$102.98`) because GSAP `innerText` interpolation animated continuous floats while both HyperFrames overlays and Remotion's `PersistentLedgerRibbon` rendered overlapping counters.
- **Decision:** Add `build_canonical_numeric_state_trace` in `motionlab/composition/continuous_timeline.py`, enforce `ALLOW_FRAC` integer-dollar locking and discrete milestone stepping (`discrete_steps_cents`) in `motionlab/composition/primitive_adapters.py` (`hf_apple_money_count`, `hf_mk_progress_stat`), and disable `show_persistent_ribbon` whenever an active scene layer owns the numeric display.
- **Consequences:** Every rendered frame agrees with the exact integer-cent ledger (`integer_dollar_lock=True`, `single_numeric_owner_verified=True`). Fractional cents appear ONLY when spoken explicitly in the script (verified by `EV06_explicit_fractional_cents`).

## ADR-03: Continuous Timeline, Shared-Element Bridges & Objective Decoded-Pixel Critic

- **Context:** R5.R1 cutaways terminated with hard cuts (`f417` MAD `114.16`, `f533` MAD `122.43`, `f635` MAD `114.40`), and `hf_organic_light_leak` caused face contrast washout at `f680`.
- **Decision:**
  1. Build Python/TypeScript mirrored continuity models (`motionlab/composition/continuous_timeline.py`, `renderers/remotion/src/composition/continuity.tsx`) implementing `EditorialSheetSweepInBridge` and `PaperFoldExitBridge` (12-frame handles keeping live source frames continuously underneath).
  2. Restrict `hf_organic_light_leak` strictly to peripheral edges (`opacity <= 0.22`) with an explicit radial center-face transparency mask.
  3. Add an independent decoded-pixel critic (`motionlab/composition/visual_critic.py`) that measures real boundary MADs on decoded MP4 frames, generates `boundary_strips.jpg`, guards against live-scene face washout, and honestly reports `vlm_status = "VISION_CRITIC_NOT_RUN"` (never treating text-only Qwen2.5-3B as a VLM).
- **Consequences:** All 10 scene boundaries dropped below threshold (`abrupt_cutaway_jumps=0`; `f417` MAD `2.76`, `f533` MAD `22.63`, `f635` MAD `3.98`), `max_face_washout_luma_delta = 9.12` (0 findings across 720 frames).

## ADR-04: Post-Render Animation Marker Barrier, Cue-Ledger Conservation & Real Counterfactual Audio QA

- **Context:** R5.R1 suffered four P0 audio defects: D09 (50ms duration gate silently dropped 18ms `data-tick` and 49.5ms `foley-card-touch`, wiping out Bills f152/f166 and renumbering `cue_id`s), D10 (gradual paper/envelope assets anchored to raw sample peak instead of envelope crest, causing `+6.0f` / `+5.38f` lag, plus `+7.03 dB` 1–4 kHz masking on `page_turn`), D11 (fake single-cue ablation reusing `snr_over_codec_db`, loose `>=85%` pass threshold, and `1,325,352` sample overhang), and D12 (audio mixed before animation markers were realized).
- **Decision:**
  1. Enforce a strict post-render dependency barrier in `r5_engine.py` (`extract_realized_animation_markers` runs AFTER HyperFrames + Remotion compilation).
  2. Lower minimum valid transient length to `8ms`, preserve immutable original `cue_id`, replace the global 16-cue cutoff with sliding-window density control (`8/10s`), and enforce mathematical conservation `planned == staged + suppressed + blocked` in `cue_decisions.json`.
  3. Resolve mixer PCM strictly by immutable `(asset_id, raw_asset_sha256)`, apply role-aware anchor alignment (`transient` vs `crest_in_window`), prefer authentic `props-page-turn` CC0 WAV for `page_turn`, carve `1.0–4.0 kHz` on sustained paper/page Foley (never phase-shifting crisp `cash_flick` transients), align master mix to exact `timeline.duration_samples` (`1,323,000`), and normalize FFT 1–4 kHz measurements via Parseval's theorem.
  4. Replace fake ablation in `validate_final_audio_mix()` with **true minus-one-cue AAC-192k counterfactual encodes** under frozen headroom gains, requiring 100% pass across all staged cues.
- **Consequences:** Canonical cold run achieved `13/13` staged (`0` dropped, `0` blocked), `13/13` true counterfactual ablation (`46.6–55.8 dB`), `13/13` role-aware sync (`+0.00f` to `-0.16f`), `13/13` speech-band clarity (`page_turn` `-7.93 dB`, improved by `14.96 dB` vs R1), `master_exact=True`, `0` clipping.

## ADR-05: Active-Window Remotion Ablation Key & Render Optimization

- **Context:** Caching `selectComposition()` in `batch_render.mjs` by `sceneId` without `excludeLayerRole` caused Remotion v4 to reuse full-scene props (`excludeLayerRole: null`) for ablation stills (`0/18` verified); HyperFrames also spawned an external CLI lint subprocess per scene.
- **Decision:** Key `compCache` by `excludeLayerRole`, enforce `ablation_layers_verified == ablation_layers_total` in `r5_engine.py`'s `tech_ok` gate, add fast in-process HTML static linting (`skip_cli_lint=True`) in `hyperframes.py`, and upgrade `motionlab validate --run` in `pipeline.py` to perform non-destructive R5.R2 artifact re-verification without clobbering `contact_sheet.jpg` or `validation.json`.
- **Consequences:** `18/18` active-window (entry/peak/exit) visual ablations pass genuinely; cold HyperFrames stage dropped `110.05s -> 88.83s`; warm-cache total time dropped `525.08s -> 230.19s` (`2.28x` speedup); `motionlab validate --run` verifies `16/16` checks non-destructively (`TECHNICAL_PASS`).

## ADR-06: Honest Product Asset Provenance & Strict ART_DIRECTED_QA Isolation

- **Context:** No owner-authorized Notion dashboard screenshot or workbook cover exists in the workspace.
- **Decision:** Create `motionlab/assets/catalog.py` enforcing SHA-256 verification, explicit owner authorization, PII redaction, and path-traversal guards. When absent, record `ASSET_MISSING_CONCEPTUAL` and render clearly labeled conceptual schematics (`CONCEPTUAL PRODUCT OVERVIEW`). Keep H1–H5 isolated Visual Lab proofs strictly under `runs/r5r2_visual_lab/` (`ART_DIRECTED_QA`) and never substitute them into canonical `AUTO_GENERATIVE`.
- **Consequences:** Zero fabricated screenshots or false provenance claims; canonical AUTO remains 100% automatic from raw MP4.

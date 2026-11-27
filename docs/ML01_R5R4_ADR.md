# Architecture Decision Record — MotionLab Studio ML-01.R5.R4

- **Repository:** `https://github.com/quocan050302/MotionLab-Studio`
- **Branch:** `feat/ml-01-zero-api-effects`
- **Base Commit (R5.R3 Docs Head):** `8c4a437bea0148e6e6234afa33a05249dc58d986`
- **Base Tested Code Commit (R5.R3):** `80d344e812fd9a2aaeef57e205e3b9dc846ef614`
- **Runtime Mode:** 100% Offline / Zero-API (Faster-Whisper Small + built-in Silero VAD ONNX, Qwen2.5-3B-Instruct Q4_K_M via `llama-completion`, MODNet ONNX, Remotion + HyperFrames + FFmpeg, local CC0-1.0 Foley library).

---

## ADR-01: Local Speech Perception, Neural Silero VAD & Conservative Word Protection (Phase A1, Fixes D03)

- **Context:** R5.R3 treated transcript word timing as `SEGMENT_APPROX` without neural VAD cross-checks, local energy boundary checks, or per-word uncertainty envelopes, making safe timeline cutting impossible.
- **Decision:** Add `motionlab/semantic/speech_perception.py` reusing the installed `faster_whisper.vad` neural Silero VAD ONNX model (zero new dependencies) paired with 20 ms spectral/energy cross-checks, 5 ms RMS envelope consonant-tail protection (`VAD_ENVELOPE_REFINED` / `ASR_WORD_ESTIMATE` / `SEGMENT_APPROX` — never fabricating `FORCED_ALIGNED`), chunked voiced F0/prosody proxies, and bounded-memory pause-window visual motion/ROI sampling (`UNKNOWN_ACTION` when no local VLM is provisioned).
- **Consequences:** Soft trailing syllables, inhalations, and VAD-spectral disagreement windows are conservatively protected; silent/nonverbal clips are explicitly detected and never misclassified as dead air.

## ADR-02: Causal Multi-Niche Intelligent Pacing Director & Fixed Semantic Eval Oracle (Phase A2, Fixes D02, D04, D07)

- **Context:** R5.R3 preserved all source frames blindly (`breathing_bridge` had no editorial decision model), `eval_harness.py::run_hybrid_eval_suite()` checked `bool(sb.storyboard_hash)` instead of validating semantic niche/facts (passing mislabeled `EV24`/`EV32`), and short allocation beats (`<2.1s`) triggered jarring full-screen `aspect_morph_editorial` flips.
- **Decision:**
  1. Add `motionlab/semantic/pacing_director.py` classifying pause candidates strictly from A1 evidence + StoryGraph (`DEAD_AIR`, `BREATH`, `EMOTIONAL`, `COGNITIVE`, `VISUAL_ACTION`, `TRANSITION`, `AMBIGUOUS`, `INTENTIONAL_SILENCE`, `END_TAIL`), proposing distinct global variants (`identity`, `conservative`, `balanced`, `tight`), scoring on information rhythm & naturalness without fixed duration quotas, and causally selecting `EditTimeMap` before downstream compilation.
  2. Expand `NICHE_PROFILES` (`tech`, `comedy`, `wellness`, `silent_art`, `mixed`, `unknown`) and reconcile clear structural niche evidence against small-LLM mislabels while recording raw 3B outputs verbatim.
  3. Replace `bool(sb.storyboard_hash)` in `eval_harness.py` with independent `evaluate_storyboard_semantics()` checking niche, factual ledger conservation, forbidden finance primitives on non-finance clips, and strict UI protection on gameplay/tutorial (`EV01..EV44`).
  4. Bias short allocation beats (`<2.1s`) toward continuous live money-world rails instead of brief full-screen chart cutaways (D07).
- **Consequences:** Reversing pacing candidate scores causally alters `EditTimeMap` and output frames/audio; wrong-niche mutations fail closed even with a valid `storyboard_hash`.

## ADR-03: Canonical EditTimeMap & Strategy A Near-Lossless Mezzanine + Crossfaded Voice Bed (Phase A3–A4, Fixes D01, D05, D06)

- **Context:** Cutting video & audio directly across multi-engine Remotion/HyperFrames/MODNet/SFX graphs risks source-vs-edited frame double-remapping, floating sample drift on rational FPS (`24/1`, `24000/1001`, `30000/1001`), and audio splice clicks.
- **Decision:** Adopt **Strategy A** via `motionlab/composition/edit_time_map.py` and `motionlab/audio/mixer.py::build_edited_voice_bed`:
  1. Construct an immutable, hash-locked `EditTimeMap` using exact integer frames and `Fraction` rational sample allocation (`output_total_frames = sum(kept_spans)` independently derived; removed frames map strictly to `None`; cuts round strictly inward and fail closed on any protected overlap).
  2. Generate a controlled near-lossless CRF-10 `yuv420p` edited mezzanine video (content-addressed cached by `(source_sha, edit_map_hash)` when cache enabled) and a sample-exact derived `edited_voice_bed.wav` with 12 ms equal-power cosine micro-crossfades strictly inside non-speech join margins.
  3. Keep the original input MP4 immutable and stream-copy `original_audio_master.m4a` bit-exact from the original source.
  4. Verify word preservation via local Faster-Whisper re-ASR on `edited_voice_bed.wav` (never mixed SFX) and run `edit_continuity.py` (`cut_continuity.json`, `join_audio_qa.json`, `cut_join_strips.jpg`); roll back suspect cuts (max 2 rounds) before compiling edited-domain scenes & SFX.
  5. Branch `r5_engine.py` and `pipeline.py` validation strictly by `--pacing preserve|auto` (default `preserve` keeps exact source frame count; `auto` validates against independently verified `EditTimeMap.output_total_frames`; `--pacing auto` + `--audio-mode preserve-exact` fails closed at config validation).
- **Consequences:** Single clean edited-domain timeline drives Remotion, HyperFrames, MODNet matte cache (content-addressed -> 9/9 warm Remotion hits), 4-tier markers, and floor-free counterfactual audio QA without double-remapping or sample drift.

## ADR-04: Objective Rhythm Critic & Privacy-Safe QA-02 Pacing Evidence (Phase A8–A10, Fixes D10, D11, D12)

- **Context:** Long-form scaling and remote review require bounded-memory evaluation of post-edit pacing directly on `edited_voice_bed.wav` plus redacted time-map/pacing telemetry in QA-02.
- **Decision:** Add `run_rhythm_critic()` to `motionlab/composition/visual_critic.py` (emitting `rhythm_critic.json` and `cut_review.json`) and extend `qa_pack.py` QA-02 to include redacted `edit_time_map.json`, `pacing_decisions.json`, `word_preservation.json`, `cut_continuity.json`, `join_audio_qa.json`, and `rhythm_critic.json` within the 2 MiB budget (`CONTACT_SHEET_LOCAL_ONLY`, `ARTIFACT_LOCAL_ONLY`, fixed `_WIN_ABS_RE` preserving `https://` URLs).
- **Consequences:** Technical and rhythm verification stay strictly separated from human VQ-01 aesthetic & listening sign-off (`PENDING_HUMAN_REVIEW`).

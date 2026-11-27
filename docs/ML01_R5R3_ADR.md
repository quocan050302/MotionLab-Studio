# Architecture Decision Record — MotionLab Studio ML-01.R5.R3

- **Repository:** `https://github.com/quocan050302/MotionLab-Studio`
- **Branch:** `feat/ml-01-zero-api-effects`
- **Base Commit (R5.R2 Docs Head):** `4aaef79a7b5f24036334d4789a6d3bfe34948f27`
- **Base Tested Code Commit (R5.R2):** `04ee4faceb107ee3f2470b52f3734c2826d37efe`
- **Runtime Mode:** 100% Offline / Zero-API (local Faster-Whisper Small, local Qwen2.5-3B-Instruct Q4_K_M via `llama-completion`, local MODNet ONNX, local Remotion + HyperFrames + FFmpeg, local CC0-1.0 Foley library).

---

## ADR-01: Causal Global Creative Director & Beam-Search Candidate Selection (Fixes D01, D02, D03, D12, D17)

- **Context:** In R5.R2, `direct_generative_storyboard()` constructed `SceneSpec` first and only ran `rank_scene_candidates()` afterward as an audit report (non-causal). Additionally, `rank_scene_candidates()` raised `IndexError` on empty candidates and returned ineligible `scored[0]` when all candidates failed hard constraints; `EvidenceIndex` rebuilt synthetic beat-wide transcripts instead of accepting the real ASR `Transcript`; retrieval was pure bag-of-words; and the director cache key omitted prompt/schema/catalog/ROI/model/binary fingerprints.
- **Decision:**
  1. Refactor `motionlab/semantic/generative_director.py` into a **strictly causal pipeline**: real `Transcript` + ROI tracks + `ProductAssetCatalog` -> `EvidenceIndex` (honest `timestamp_quality`: `WORD_ALIGNED_APPROX` | `SEGMENT_APPROX` | `BEAT_LEVEL_APPROX`) -> `NarrativeContract` -> `NarrativeStateGraph` -> `ontology_hybrid_sparse` retrieval (`catalog_search.py`) -> local LLM JSON pass with canonical **D17** cache key (`model_profiles.py::compute_director_cache_key`) -> multi-candidate proposals -> **Global Coherent Beam Search** (`rank_global_candidates`, beam width 8, cross-beat rhythm/diversity/bridge scoring + bounded replan) -> direct compilation of `SelectedScenePlan` into `SceneSpec`/`LayerSpec`/`TransitionSpec`/`SoundIntent` -> locked `selected_graph_hash` + `storyboard_hash`.
  2. Harden `rank_scene_candidates()` to fail closed with `NO_ELIGIBLE_CANDIDATE` sentinel on empty or all-rejected candidate sets and remove `_optA_` / blind dual-engine scoring biases.
  3. Expand `motionlab/semantic/eval_harness.py` from 16 to **36 independent cases (`EV01..EV36`)** with `run_hybrid_eval_suite()` separating deterministic rules vs raw LLM vs reconciled overrides by intervention type (`safety`, `factual`, `artistic`, `resource`).
- **Consequences:** Changing candidate rankings causally alters the compiled storyboard primitives, layouts, transitions, sound intents, `selected_graph_hash`, and `storyboard_hash`. Cache entries automatically invalidate on any semantic, schema, prompt, catalog, ROI, model, or binary change.

## ADR-02: Multi-HyperFrames-Layer Compiler IR, Zero Hardcodes & Rational FPS Clock (Fixes D04, D05, D13)

- **Context:** R5.R2's `compile_scene_hyperframes_html()` compiled only the first HyperFrames layer and silently dropped subsequent layers; `GenerativeSceneComposition.tsx` and `primitive_adapters.py` still held Olivia-specific fallback strings; and `compile_scene_remotion_props()` silently rounded `fps_num / fps_den` to `int(round(...))` without NTSC support or fail-closed rejection of unsupported rates.
- **Decision:**
  1. Introduce `CompiledOverlay` and `compile_scene_hyperframes_layers()` in `motionlab/composition/generative_compiler.py` compiling ALL HyperFrames layers sorted by `z_order` (rejecting duplicate `layer_id`s), plus `render_mode` (`combined` | `paper_tokens_only` | `typography_only`) in `hf_apple_money_count` for genuine 2-HF-layer depth compositing behind and above MODNet subject mattes.
  2. Upgrade `GenerativeSceneComposition.tsx` and `batch_render.mjs` to render `hfLayers[]` in exact `zOrder` and support per-layer `excludeLayerId` ablations (`compCacheKey` includes `excludeLayerId`).
  3. Strip all Olivia-specific fallback strings from runtime TSX, HyperFrames adapters, and primitive registry tags.
  4. Add `validate_rational_fps_clock()` supporting integer rates (`EXACT_INTEGER`) and standard NTSC rates (`24000/1001`, `30000/1001`, `60000/1001` as `RATIONAL_NTSC_RESAMPLED`) while failing closed on unsupported irrational frame rates.
- **Consequences:** Multi-HF-layer scenes execute and verify every individual layer via per-layer pixel ablation; empty-parameter adapters never leak Olivia text; unsupported rational clocks fail closed before render.

## ADR-03: Analytic Physically-Inspired Motion Language (A6)

- **Context:** Continuous motion across scenes needs deterministic, non-exploding, material-aware trajectories synchronized between Python planning and TypeScript Remotion rendering without falsely claiming a rigid-body physics simulator.
- **Decision:** Add `evaluate_motion_state()`, `sample_motion_path()`, `build_material_motion()`, `resolve_contact_event()`, and `validate_track_continuity()` to `motionlab/composition/continuous_timeline.py` with a matching TypeScript mirror `evaluateMotionStateTS()` in `renderers/remotion/src/composition/continuity.tsx`. Every output explicitly sets `motion_model = "analytic_physically_inspired"`.
- **Consequences:** Shared elements follow smooth anticipation -> acceleration -> peak velocity -> settle trajectories with finite velocity/acceleration/flex bounds and verifiable release/peak/contact markers.

## ADR-04: Audio Asset Intelligence, 4-Tier Marker Truth & Floor-Free Counterfactual Mix QA (Fixes D07, D08)

- **Context:** R5.R2 labeled compiled-prop animation markers as `"realized"` without verifying rendered pixels, and `validate_final_audio_mix()` applied an artificial `cue_diff_rms * 0.5` floor that could mask a missing cue in `final.mp4` when `mixed_reference.wav` was intact.
- **Decision:**
  1. Add `build_audio_asset_intelligence()` in `motionlab/audio/assets.py` deduplicating unique raw WAV SHA-256 hashes and computing real FFT spectral/envelope descriptors (`spectral_centroid_hz`, band ratios, `decay_ms`, `crest_factor_db`, `roughness_proxy`), honestly reporting `audio_embedding_model_used=False`.
  2. Implement a strict 4-tier `AnimationMarker` provenance protocol (`PLANNED` -> `COMPILED` -> `RENDER_EMITTED` -> `OBSERVED_PIXEL` / `OBSERVED_AUDIO`) in `motionlab/audio/semantic_sound.py`, upgrading to `OBSERVED_PIXEL` only after measuring actual decoded pixel delta around the marker window.
  3. Remove the artificial `cue_diff_rms * 0.5` floor in `motionlab/audio/mixer.py` (`reference_floor_used=False`). Check A verifies reference identifiability; Check B directly aligns decoded `final.mp4` using cue-exterior leading+trailing context (safe even at frame `t≈0`) against both intact and minus-one controls, requiring `final_ablation_drop_db >= 6.0`, `fin_minus_in_rms < fin_minus_cf_rms * 0.72`, and `residual_correlation >= 0.45`.
- **Consequences:** Tampering `final.mp4` (replacing with source-only audio while keeping `mixed_reference.wav` intact) fails closed with `final_presence_verified=False`; genuine mixes pass with 100% floor-free verification.

## ADR-05: Strict Authorized Product Grounding (Fixes D06)

- **Context:** Weak substring matching (`"" in s`) and fake file headers risked matching unrelated/corrupt files as authorized product assets.
- **Decision:** Harden `motionlab/assets/catalog.py` to decode real image/video bytes (`>=64x64`), enforce extension allow-lists and path containment, match strictly by normalized exact `product_name` / `product_id` / `approved_aliases`, reject ambiguous multi-product matches (`AMBIGUOUS_PRODUCT_MATCH`), and mount authorized textures directly into `rem_rc_flip_page` (`<Img>` / `<OffthreadVideo>`) with `product_asset` ablation verification (`AUTHORIZED_DEMO_RENDERED_VERIFIED`). When absent, retain honest `ASSET_MISSING_CONCEPTUAL`.
- **Consequences:** Real authorized assets render cleanly inside the 3D page flip (proven in Visual Lab H4 with pixel ablation); missing/unauthorized/corrupt assets never fake product UI.

## ADR-06: Streaming O(window) Objective Critic, Honest VLM Adapter & Bounded Repair Loop (Fixes D09, D10, D11)

- **Context:** R5.R2 buffered all decoded frames in RAM (`fin_frames`, `src_frames`), only inspected exact single-frame jumps at scene boundaries (missing internal flashes and post-boundary spikes), and hardcoded `VISION_CRITIC_NOT_RUN` without probing `models/manifest.json`.
- **Decision:**
  1. Rewrite `motionlab/composition/visual_critic.py` using a streaming `deque(maxlen=16)` ring buffer (O(window) RAM independent of clip length) scanning ALL frames for internal single-frame flashes, black frames, face-contrast washout, and ±6f boundary-window spikes (`post_4f_source_relative_mad`, `severe_findings_count`).
  2. Add `inspect_local_vlm()` / `run_local_vlm_critic()` probing `models/manifest.json` + local multimodal binaries (`llama-mtmd-cli` / `llama-llava-cli`), honestly reporting `VISION_CRITIC_NOT_RUN` when no verified local VLM is provisioned.
  3. Wire a **Bounded Creative Repair Loop** (max 2 rounds) in `r5_engine.py` before freezing markers and audio mixing: only affected scenes with `HIGH` findings are patched and rerendered, rolled back if unimproved, and logged in `evidence/revision_history.json`.
- **Consequences:** Bounded memory scales cleanly to long-form video (`1440+` frames), internal flashes and post-boundary spikes fail closed unless repaired, and no text-only LLM is ever misrepresented as a VLM.

## ADR-07: Hardened Technical Gates & Atomic Safe Redacted QA-02 Pack (Fixes D14, D15, D16)

- **Context:** `tech_ok` previously treated empty ablation sets (`0/0`) or missing `mix_info` as passing, and QA pack assembly used direct `shutil.rmtree` without containment checks or complete redaction of free-text paths/quotes/emails.
- **Decision:**
  1. Harden `r5_engine.py` and `pipeline.py` so `tech_ok` requires `ablation_total > 0` whenever planned graphic/matte layers exist, `mix_info is not None` whenever mixed audio cues are planned, `severe_findings_count == 0`, and all scenes backed by valid `chosen_candidate_id`.
  2. Rewrite `motionlab/composition/qa_pack.py` with strict `ensure_safe_subpath` containment under `allowed_root`, zero blind `shutil.rmtree`, atomic staging + swap, recursive redaction of quotes, POSIX/Windows host paths, and emails, plus explicit `image_sharing_authorized` gating (`CONTACT_SHEET_LOCAL_ONLY` / `ARTIFACT_LOCAL_ONLY` / `url=null` by default).
- **Consequences:** Zero silent vacuous passes and zero leakage of local paths, verbatim transcripts, PII, or unauthorized screenshots into Git-tracked QA-02 packs.

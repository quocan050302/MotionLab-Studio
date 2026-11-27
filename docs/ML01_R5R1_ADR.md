# Architecture Decision Record — MotionLab Studio ML-01.R5.R1
**Premium Art Direction, Two-Pass Semantic Sound Director, CC0-1.0 Foley Provisioning & Single-Browser Render Optimization**

- **Date:** 2026-11-27
- **Base Lineage:** `feat/ml-01-zero-api-effects` descending cleanly from `303df5bf` (R5 tested code `0842f4ee`) and upstream `browser-use/video-use` (`43cfc566`, MIT)
- **Invariant Scope:** Zero runtime cloud/paid APIs, original input immutable, voice master bit-exact (`1296` packets), strict separation between `AUTO_GENERATIVE` and `ART_DIRECTED_QA`, 20 verified imported mechanisms preserved untouched.

---

## ADR-R5R1-01: Open Editorial Art Direction & All-Frame Safe-Zone Solver (`motionlab/composition/art_direction.py`)
- **Problem (R5 Defects D01–D05, D09):** R5 wrapped almost every overlay in identical dark rounded espresso cards (`border-radius: 24-26px`), leaked internal `"(derived)"` provenance text into consumer captions, used flat green `$` rectangles, transitioned into Payoff through a dark espresso iris (`f348`), left an empty cream gap in `aspect_morph_editorial`, and lacked an all-frame Reels safe-zone & eye/mouth sweep.
- **Decision:**
  1. Adopt cohesive warm editorial tokens (`#F5F0E6` ivory, `#84977A` sage, `#243B32` deep forest, `#40382E` muted espresso, `#BE9A59` antique gold, `#AF745E` terracotta) derived via `build_style_tokens(niche, source_mp4)`.
  2. Replace generic dark rounded pill cards with open borderless editorial typography + subtle top atmospheric gradient scrims on live footage, warm-ivory architectural strips for allocation counters, seamless borderless open ivory cutaways for Payoff (`148px` deep forest `$100`) and Proportions, and luminous warm ivory `FrameMorphBridge` (eliminating dark circular occlusion).
  3. Upgrade `hf_apple_money_count` tokens to curated 3–4 multi-layered linen-paper banknotes with subtle 3D perspective tilt and curved entry paths.
  4. Enforce conservative 9:16 Reels safe margins (`L>=48px, R>=105px, T>=110px, B>=230px` at `720x1280`), all-frame eye/mouth collision sweep (`evidence/frame_protection.csv`), `SharedElementState` continuity tracks (`continuity.json`), and fail-closed consumer QA-jargon rejection (`(derived)`, `derived_arithmetic`, `verbatim_spoken`) in both `evaluate_phone_legibility()` and `validate_storyboard_spec()`.

## ADR-R5R1-02: Two-Pass Semantic Sound Director & Animation-Marker Alignment (`motionlab/audio/semantic_sound.py`)
- **Problem (R5 Defect D06):** R5 selected SFX from a shallow `speech_act -> family` lookup at fixed `sf+5` offsets, with no explicit `SoundRole` taxonomy, no rejected-cliché rationale, no `QuietSpan` records, and no coupling to actual rendered animation keyframes.
- **Decision:**
  1. Introduce `extract_rendered_animation_markers()` emitting frame-exact `AnimationMarker` records (`transition_peak`, `reveal_start`, `release`, `impact`, `counter_settle`, `page_contact`, `resolve`) directly from compiled Remotion & HyperFrames keyframe parameters.
  2. Introduce `extract_sonic_opportunities()` + `plan_sonic_score()` + fail-closed `validate_semantic_sound_grounding()` enforcing `SoundRole` enums (`DIEGETIC_FOLEY`, `ILLUSTRATIVE_FOLEY`, `GRAPHIC_MOTION`, `COGNITIVE_EMPHASIS`, `TRANSITION_BRIDGE`, `PRODUCT_TEXTURE`, `INTENTIONAL_SILENCE`), explicit `rejected_candidates` (rejecting cash on negated clauses, cash on non-monetary `56-page` counts, engine sound on financial `"gas"`, and casino/notification clichés), and explicit `QuietSpan` entries (natural breathing pause & final CTA intelligibility).
  3. Convert markers to exact `Fraction` sample clocks via `align_sound_to_marker()`, ensuring any animation retime shifts aligned SFX samples automatically.

## ADR-R5R1-03: Verified CC0-1.0 48 kHz 24-Bit Local Foley Pack (`scripts/provision_cc0_foley.py`)
- **Problem (R5 Defect D07):** R5 had 0 local WAV assets (`PARTIAL_FOLEY_ASSETS`), relying 100% on local Python DSP synthesis.
- **Decision:**
  1. Provision a curated 29-file (34 family/variant bindings) **CC0 1.0 Universal (Public Domain)** 48 kHz 24-bit WAV pack from `wyzowl/sfx-kit` (pinned commit `92780e7ab0cdfb557497a988089c47f46e8c7412`, verbatim `LICENSE.wyzowl-sfx-kit.txt` preserved) via one-time authorized bootstrap (`scripts/provision_cc0_foley.py`).
  2. Upgrade `load_recorded_foley_pcm()` to decode 16/24/32-bit PCM WAV and polyphase-resample (`scipy.signal.resample_poly`) to exact target sample rate using upstream `sync.at` transient offsets, while keeping WAV binaries strictly gitignored (`assets/sfx_foley/*.wav`) and falling back honestly to `PARTIAL_FOLEY_ASSETS` (`procedural_dsp_synthesis`) if local WAVs are absent.
  3. Upgrade `validate_final_audio_mix()` with per-cue peak lag (frames), `1–4 kHz` speech-band masking ratio, mono-downmix compatibility check, and single-cue ablation verification, while explicitly preserving `LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY` whenever no human playback review has occurred.

## ADR-R5R1-04: Single-Browser Remotion Batch Worker, Content-Addressed Caches & Active-Window Ablation (`renderers/remotion/scripts/batch_render.mjs`)
- **Problem (R5 Defects D08 & D10):** R5 invoked `remotion.cmd` 33 separate times (9 scenes + 24 midpoint stills = ~330s) with zero per-scene/per-overlay render cache, and midpoint-only ablation missed entry wipe transitions (`13/15`).
- **Decision:**
  1. Build `renderers/remotion/scripts/batch_render.mjs` using already-installed local `@remotion/bundler` and `@remotion/renderer` (`4.0.547`): content-addressed Webpack bundle cache keyed by exact TSX/dependency `code_hash`, hardlinked isolated per-scene `public/<scene_id>/` mounts, and ONE persistent `openBrowser()` Chromium instance across all uncached scenes and all active-window ablation stills (closed cleanly in `finally`).
  2. Add content-addressed per-scene Remotion cache (`remotion_scenes/rem_<hash>.mp4`) and per-overlay HyperFrames cache (`hyperframes_overlays/hf_<hash>.webm`) keyed by engine version, code/HTML hash, normalized props, media/matte/HF SHA-256s, dimensions, FPS, codec, and seed (bypassed cleanly on `--no-cache`).
  3. Replace midpoint-only ablation with **Active-Window Ablation (`entry / peak / settle`)**: entrance transitions verified at `transition_in.duration_frames // 2`, 3D page flips verified at mid-flip, and sustained layers verified at scene midpoint inside the same single-browser batch.

## ADR-R5R1-05: QA-01 Lightweight Redacted Evidence Packs (`motionlab/composition/qa_pack.py`)
- **Decision:** Emit strictly `<= 2 MiB` Git-safe packs under `qa_evidence/ML-01.R5.R1/<tested_code_sha_short>/<run_id>/` containing relative paths only, redacted verbatim transcript hashes (no unapproved public transcript/footage publication), exact artifact hashes (`ARTIFACT_LOCAL_ONLY`), and zero MP4/WAV blobs in Git history.

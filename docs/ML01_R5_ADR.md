# Architecture Decision Record — MotionLab Studio ML-01.R5.REV2

**Status:** Implemented & Verified on Windows 11 x64 (`feat/ml-01-zero-api-effects`)  
**Date:** 2026-11-27  

## ADR-01: Safe Dual-Lineage Reconciliation Without Destructive Git History
- **Context:** Remote `origin/feat/ml-01-zero-api-effects` (`f374b7e`) was a newer Windows-hardened ML-01 signal-only lineage, while historical R4 (`6bcdb94`) held semantic ASR/LLM, MODNet vision, full-frame Remotion/HyperFrames, and audio mixing on a diverged branch sharing upstream `43cfc56`.
- **Decision:** Create local staging branch `integration/ml01-r5-20261127` strictly descending from `f374b7e`, mount `6bcdb94` read-only, preserve all Windows hardening/40 effects/tests from HEAD, and port R2–R4 capabilities as clean compatibility patches (unified `RationalFraction`, recursive process-tree RSS tracking, FFmpeg 9.0.2 `-fps_mode passthrough`, CPU `-ngl 0` for Windows `llama.cpp`). Publish strictly via fast-forward (`git merge-base --is-ancestor f374b7e HEAD`).
- **Consequences:** Zero lost commits, zero force-push, full backward compatibility with legacy ML-01/R2/R3/R4 invocations.

## ADR-02: Repository Reuse First + Unified Primitive Registry (`VERIFIED` Gate)
- **Context:** Hand-coding bespoke SVG cards/rings caused R4's "technical PASS, visual FAIL" problem (tiny 14px text, fake index-based bar widths, repetitive rounded-box webcam zooms).
- **Decision:** Audit 20 upstream repos, verify exact root `LICENSE` files + commit SHAs, import **20 distinct mechanisms** across **5 MIT/Apache-2.0 repositories** (`heygen-com/hyperframes`, `riaz37/remotion-ui`, `vivmagarwal/remotion-effects-library`, `RenderComp/free-remotion-templates`, `GTKottman/remotion-transitions`) into `renderers/{remotion,hyperframes}/imported/`, and lock SHA-256 before/after in `third_party/motion_sources.lock.json`.
- **Lifecycle Gate:** Every imported mechanism must pass real upstream smoke + adapted render verification (`third_party/imported_primitive_manifest.json`) before `UnifiedPrimitiveRegistry.verified_primitives()` exposes it to the AI director. Canonical outputs must pass `validate_canonical_reuse_gate` (>=4 verified imported mechanisms across BOTH Remotion and HyperFrames, >=2 per engine).

## ADR-03: Integer-Cent Verified Financial Ledger & StoryGraph (`story_graph.py`)
- **Context:** R4's per-unit numeric grounding blocked hallucinations well, but could not carry derived intermediate balances (`$1,200 - $800 = $400`) across consecutive spoken clauses or isolate non-monetary counts (`56-page workbook`) cleanly.
- **Decision:** Parse all monetary quantities strictly into integer cents (`int`), maintain a running-balance state machine separating `provenance="verbatim_spoken"` from `provenance="derived_arithmetic"`, enforce exact conservation (`initial - sum(allocations) == computed_remaining`), handle reordered categories/currencies/overspend cleanly, isolate negated statements (`doesn't mean you have $1,200 to spend`), and reject hallucinated savings promises (`SAVE $1,200`, `guaranteed profit`) fail-closed.

## ADR-04: Safe Typed Generative DSL Over Unrestricted Code Generation
- **Context:** Allowing an offline 3B LLM to emit arbitrary executable JSX/JS/HTML risks syntax failures, XSS/injection, remote asset fetches, and runaway particle/layer counts.
- **Decision:** Local `Qwen2.5-3B-Instruct` (`llama-completion.exe --json-schema-file`, 100% offline, seeded) searches the verified primitive registry via `search_primitives_by_intent` and outputs a typed `StoryboardSpec` JSON combining stage modes, imported Remotion & HyperFrames primitives, transitions, and sound intents. `validate_storyboard_spec()` enforces strict caps (`<=24` layers/scene, `<=1500` keyframes, `<=120` particles), finite numbers, monotonic non-overlapping frame clocks, zero forbidden tokens (`http://`, `<script`, `eval(`, `../`), and verified claim grounding. Compiled deterministically to `MotionLabGenerativeScene` + offline HyperFrames GSAP HTML (`./gsap.min.js`).

## ADR-05: Voice-Priority Per-Event Audibility Mixing & Matched No-SFX Codec QA
- **Context:** R4's SFX stem peaked at `-20.05 dBFS` with blanket `-4.5 dB` speech ducking, rendering cues nearly inaudible, while simple residual RMS mixed codec quantization noise with SFX energy.
- **Decision:**
  1. Stream-copy bit-exact `original_audio_master.m4a` (1296 packets for Olivia) untouched.
  2. Keep voice at unity gain `1.0` (zero EQ/NR/compression/normalization).
  3. Calibrate every SFX cue against nearby local voice RMS with a targeted short sustain tuck (`15ms` attack / `110ms` release, preserving crisp 35ms transients) and local SFX-only headroom guard keeping final decoded true peak `<= -1.0 dBTP` with `0` added clipping.
  4. Generate a matched no-SFX AAC-192k reference (`voice_only_matched.m4a`) to isolate true per-event SFX SNR over the AAC codec floor.
  5. Honestly label local physical-modeling synthesis as `procedural_dsp_synthesis` and report `PARTIAL_FOLEY_ASSETS` when no owner-supplied CC0 recorded WAVs exist in `assets/sfx_foley/`.

## ADR-06: Strict Art-Directed Benchmark vs Canonical Automatic Generalization Separation
- **Decision:** Keep the Section 09 Olivia art-directed storyboard strictly inside `scripts/render_olivia_benchmark.py` writing to `runs/olivia_ml01_r5_benchmark/<run_id>/art_directed_benchmark.mp4`. Canonical CLI `--composition generative` NEVER imports, reads, or branches on Olivia fixtures, filenames, or storyboards.

## ADR-07: Radical Honesty in Gates & Local Resource Governance
- **Decision:** Separate technical verification (`TECHNICAL_PASS`) from human aesthetic approval (`G8: PENDING_HUMAN_REVIEW`). Never treat text-only Qwen2.5-3B as a vision critic (`VISION_CRITIC_NOT_RUN`), never claim competitor parity (`REFERENCE_NOT_AVAILABLE`), and serialize browser-heavy Chromium jobs (`--concurrency=1`, `--low-memory-mode`, `--bundle-cache`) while bounding local LLM threads (`-t 6 -ngl 0`) to protect 7–9 GiB free Windows RAM.


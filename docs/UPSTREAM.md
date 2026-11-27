# Upstream Provenance — MotionLab AI (ML-01)

This repository (`MotionLab-AI-ML01`) is an independent, local-first, zero-API greenfield project forked/cloned from upstream [`browser-use/video-use`](https://github.com/browser-use/video-use).

## Upstream Snapshot Metadata

- **Upstream Repository:** `https://github.com/browser-use/video-use.git`
- **Remote Name:** `upstream` (read-only by convention; never push to `upstream`)
- **Cloned Commit SHA:** `43cfc566833548093455c9d4a46bd69913b7e552`
- **Upstream Commit Author:** Gregor Žunič
- **Upstream Commit Date:** `2026-10-08T15:16:59-07:00`
- **Upstream Commit Subject:** `README: point cloud link to video-use.com waitlist (#222)`
- **Clone Timestamp (UTC):** `2026-11-27T00:55:12Z`
- **Local Development Branch:** `feat/ml-01-zero-api-effects`
- **Upstream License:** MIT License (`LICENSE` preserved verbatim with original copyright `Copyright (c) 2026 Browser Use`)

## Upstream Audit Map

Audited all upstream files at SHA `43cfc566833548093455c9d4a46bd69913b7e552`:

| Upstream Path | Purpose in Upstream | Status in ML-01 |
|---|---|---|
| `README.md` | Agent-driven Claude Code + ElevenLabs workflow docs | Updated to local-first ML-01 quickstart; upstream docs noted as legacy |
| `LICENSE` | MIT License (Copyright 2026 Browser Use) | Preserved verbatim |
| `install.md` | Agent skill install requiring ElevenLabs API key & `brew`/`ln -sfn` | Retained as legacy reference; NOT executed by ML-01 |
| `SKILL.md` | LLM conversation editing skill | Retained as legacy reference; NEVER loaded in ML-01 runtime |
| `.env.example` | `ELEVENLABS_API_KEY=` | Updated to document local-first (no API key required) with legacy opt-in comment |
| `pyproject.toml` | Package config (`requests`, `librosa`, `matplotlib`, `pillow`, `numpy`) | Extended for `motionlab` CLI package while preserving upstream metadata |
| `helpers/transcribe.py` | ElevenLabs Scribe cloud upload (`requests.post`) | **ISOLATED LEGACY** — never imported or called by `motionlab` |
| `helpers/transcribe_batch.py` | Parallel ElevenLabs Scribe batch uploader | **ISOLATED LEGACY** — never imported or called by `motionlab` |
| `helpers/pack_transcripts.py` | Packs Scribe JSON transcripts into markdown | **ISOLATED LEGACY** — never imported or called by `motionlab` |
| `helpers/render.py` | EDL cut renderer (applies 30ms audio fades, re-encodes AAC, loudnorm, burns subtitles) | **RETAINED LEGACY** — upstream unit tests preserved & kept passing; ML-01 uses new non-destructive `motionlab/renderers/ffmpeg.py` so source audio master and full duration are never altered |
| `helpers/grade.py` | FFmpeg `signalstats` auto-grade & presets | Retained legacy reference; ML-01 implements bounded deterministic optical effects (`V01`–`V06`) |
| `helpers/timeline_view.py` | Filmstrip + waveform PIL drilldown PNG | Retained legacy reference |
| `skills/manim-video/` | Manim CE skill docs & references | Retained legacy reference; optional Manim adapter gated off critical path |
| `tests/test_render_captions.py` | Upstream unit tests for `helpers/render.py` captions | Preserved unmodified; executed in baseline & regression suites |
| `tests/test_render_fps.py` | Upstream unit tests for `helpers/render.py` FPS parsing/probing | Preserved unmodified; executed in baseline & regression suites |
| `tests/test_render_orientation.py` | Upstream unit tests for `helpers/render.py` portrait/rotation detection | Preserved unmodified; executed in baseline & regression suites |

## Architecture Decision Records (ADRs)

### ADR-001: Dedicated Non-Destructive Renderer vs. Upstream `helpers/render.py`
Upstream `helpers/render.py` was built for cut-based EDL assembly: it extracts segments with mandatory 30ms audio fades (`afade`), re-encodes audio to AAC 48kHz, runs loudness normalization (`loudnorm`) by default, scales everything to 1080p, and burns SRT captions. ML-01 requires a full-length non-destructive timeline with a **hard audio preservation lock** (preferring bit-exact audio stream-copy, zero filters/fades/loudnorm/retime, zero default subtitles). Therefore, `helpers/render.py` is preserved untouched for legacy compatibility and upstream regression tests, while `motionlab/renderers/ffmpeg.py` implements the non-destructive ML-01 timeline renderer.

### ADR-002: Zero-API Offline Runtime Isolation
None of `motionlab/*` imports `helpers/transcribe.py`, `helpers/transcribe_batch.py`, `requests`, `urllib`, `http.client`, or any cloud SDK. All signal analysis (scene cuts, optical-flow/frame-diff motion, audio RMS/onset envelope, optional local OpenCV Haar face tracking, existing-effect protection) executes 100% locally on CPU.

### ADR-003: Multi-Engine Architecture with License Governance & Certified Fallbacks
- **FFmpeg Engine (`ffmpeg`)**: Always available locally (GPLv3/LGPL static binary via system install). Implements all 40 effects (`C01`–`C10`, `L01`–`L10`, `T01`–`T08`, `V01`–`V06`, `G01`–`G06`) via streaming FFmpeg decode/encode pipes with deterministic sub-pixel OpenCV/NumPy/Pillow frame evaluation and native FFmpeg filtergraph/compositing passes. Zero full-video RAM buffering.
- **Remotion Engine (`remotion`)**: Locked local React/TypeScript project in `renderers/remotion/`. Governed by Remotion's license (free for individuals, small teams ≤3 people, non-profits, evaluators; paid Company License for for-profit orgs ≥4 people). Opt-out via `--disable-remotion` or `MOTIONLAB_ALLOW_REMOTION=0`; automatic certified fallback to `ffmpeg` in normal mode (`BLOCKED`/error in `--strict` mode if explicitly forced).
- **HyperFrames Engine (`hyperframes`)**: Locked local HTML/CSS/GSAP project in `renderers/hyperframes/` (HyperFrames Apache-2.0, GSAP Standard No-Charge License). Invoked solely via pinned local `node_modules/.bin/hyperframes` (never `npx --yes` at runtime). Automatic certified fallback to `ffmpeg` if unavailable or disabled.
- **Manim Engine (`manim`)**: Optional gated adapter (`motionlab/renderers/manim.py`), disabled by default and off the ML-01 critical path.

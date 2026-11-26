# MotionLab-AI — ML-01 (Zero-API Automatic Video Effects Engine)

> **Standalone greenfield engine forked from [`browser-use/video-use`](https://github.com/browser-use/video-use) (`43cfc566833548093455c9d4a46bd69913b7e552`).**
> Put a raw MP4 (with your character, footage, and voice already recorded) into `input/`, run **one local offline command**, and get a technically validated `final.mp4` with restrained, rhythmic visual effects — **0 USD per-edit API cost, zero cloud calls, zero LLM/agent required at runtime, zero voice alteration.**

---

## Core Guarantees & Scope

- **100% Local Offline Runtime**: No ElevenLabs, OpenAI, Anthropic, Google, stock-download, or cloud-render calls during editing. Works with zero `.env` and zero API keys.
- **Hard Audio-Master Preservation**: Original voice/audio is never retimed, cut, faded, noise-reduced, loudness-altered, or replaced. Stream-copied (`-c:a copy`) whenever MP4-compatible (verified by raw packet SHA-256 hash) or disclosed high-bitrate AAC transcode with PCM cross-correlation validation when source uses raw PCM.
- **No Unwanted Defaults**: Zero auto subtitles, zero default background music, zero default SFX, zero watermarks, zero fabricated content or stock footage.
- **40 Distinct Rule-Based Local Effects**:
  - Camera & Composition (`C01`–`C10`)
  - Mask, Reveal & Layout (`L01`–`L10`)
  - Duration-Preserving Shot Transitions (`T01`–`T08`)
  - Color & Optical Polish (`V01`–`V06`)
  - Peripheral Motion Graphics & Audio-Reactive (`G01`–`G06`)
- **Multi-Engine Architecture**:
  - **FFmpeg + Frame-Exact Procedural Compositor** (primary, always available offline)
  - **HyperFrames** (`0.9.98`, Apache-2.0) + GSAP (`3.15.1`) for verified transparent WebM alpha overlays
  - **Remotion** (`4.0.547`, license-gated opt-in via `--enable-remotion`; see `THIRD_PARTY_NOTICES.md`)
  - **Manim** (optional, disabled by default at ML-01)
- **Honest Limitations**: ML-01 is a deterministic rule + media-signal engine (scene cuts, optical flow, audio RMS/STFT envelope, bundled OpenCV Haar face safety, UI edge heuristics). It does **not** semantically understand spoken words or story intent, and technical `PASS` does not replace human aesthetic sign-off (`PENDING_HUMAN_REVIEW`).

---

## Quick Start

### 1. One-Time Online Bootstrap

**Windows PowerShell** (non-admin, process-scoped policy, no global PATH mutation):

```powershell
cd MotionLab-AI-ML01
powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\bootstrap.ps1
# Optional Node renderers:
# powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\bootstrap.ps1 -InstallHyperFrames
```

**Linux / macOS**:

```bash
cd MotionLab-AI-ML01
bash scripts/bootstrap.sh
# Optional Node renderers:
# bash scripts/bootstrap.sh --with-hyperframes
# bash scripts/bootstrap.sh --with-remotion   # Check THIRD_PARTY_NOTICES.md license terms first
```

### 2. Verify Local Toolchain (`doctor`)

```bash
# Linux / macOS
.venv/bin/python -m motionlab doctor

# Windows PowerShell
.\.venv\Scripts\python.exe -m motionlab doctor
```

### 3. One-Command Offline Edit

Place your raw MP4 in `input/raw.mp4` (your source file is **never modified**):

```bash
# Linux / macOS
.venv/bin/python -m motionlab edit --input "input/raw.mp4" --profile auto --offline --output "runs"

# Windows PowerShell
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --offline --output "runs"
```

Open `runs/<run_id>/final.mp4`, `runs/<run_id>/contact_sheet.jpg`, and `runs/<run_id>/README.md`.

---

## CLI Reference

| Command | Description |
|---|---|
| `python -m motionlab doctor` | Inspect FFmpeg, ffprobe, Python packages, Node renderers, licenses, and free disk space |
| `python -m motionlab analyze --input input/raw.mp4 --output runs` | Run local offline media analysis (`analysis.json`) |
| `python -m motionlab plan --input input/raw.mp4 --profile auto --seed 42` | Generate deterministic `edit_plan.json` + `plan_explain.json` without rendering |
| `python -m motionlab edit --input input/raw.mp4 --profile auto --seed 42 --output runs` | Full offline edit, technical validation, contact sheet, and atomic `final.mp4` promotion |
| `python -m motionlab edit --input input/raw.mp4 --profile talking-head --aspect 9:16 --quality preview` | Explicit 9:16 preview render (fit/pad by default, never blind crop) |
| `python -m motionlab edit --input input/raw.mp4 --profile neutral --offline --strict` | Strict fail-closed mode |
| `python -m motionlab render --plan runs/<id>/edit_plan.json --input input/raw.mp4` | Replay a saved deterministic plan |
| `python -m motionlab validate --run runs/<run_id>` | Re-run machine technical checks on an existing run |
| `python -m motionlab effects list` | List all 40 registered effects, milestones, layers, and engines |
| `python -m motionlab effects gallery --output runs` | Render procedural before/after visual verification sheet for all 40 effects |
| `python -m motionlab cache prune` | Safely clear MotionLab-owned analysis cache entries |

Available explicit profiles: `auto`, `neutral`, `talking-head`, `product`, `storytelling`, `gameplay`, `tutorial`, `cinematic`.

---

## Upstream Legacy Workflow (Strictly Opt-In / Isolated)

The original upstream `browser-use/video-use` agent helpers (`helpers/transcribe.py`, `helpers/render.py`, `SKILL.md`, `install.md`) remain intact as legacy reference and are **never imported or executed by `motionlab`**. Upstream MIT license and copyright are preserved in [`LICENSE`](./LICENSE). See [`docs/UPSTREAM.md`](./docs/UPSTREAM.md), [`docs/ADR.md`](./docs/ADR.md), and [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for full provenance and third-party license details.

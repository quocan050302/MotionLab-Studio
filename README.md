# MotionLab AI — ML-01 (Zero-API Automatic Video Effects Engine)

**MotionLab AI (ML-01)** is an independent, local-first, zero-API automatic video effects CLI engine built atop upstream [`browser-use/video-use`](https://github.com/browser-use/video-use) (SHA `43cfc566833548093455c9d4a46bd69913b7e552`, MIT License).

Drop a raw MP4 (with existing visuals and voice/audio) into `input/`, run **one local CLI command**, and receive an edited MP4 with restrained, signal-timed visual effects, full technical validation, contact sheet, and untouched source audio — **100% offline with $0 per-edit API cost**.

> **Important Scope & Honesty Notice**
> - ML-01 is a **deterministic rule- and signal-driven** video effects engine (scene cuts, optical-flow motion, local PCM energy/onsets, optional local OpenCV Haar face tracking). It is **NOT** an LLM/semantic video understanding agent.
> - **Zero runtime APIs:** No ElevenLabs, OpenAI, Anthropic, Gemini, LLM, cloud render, or sub-agent is invoked during editing. No `.env` or API key is required.
> - **Operating cost:** $0 API usage fees per edit; normal local electricity, hardware, and applicable renderer licenses (see [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)) still apply.
> - **Audio hard lock:** Source audio master is preserved via stream-copy whenever MP4-compatible (zero retiming, zero voice replacement/cloning, zero fades/loudnorm/filters, zero default subtitles, zero default music/SFX).
> - **Aesthetic review:** Automated checks validate technical integrity (`TECHNICAL_PASS`), while aesthetic taste/pacing remains human review (`AESTHETIC_PENDING`). Rule-based heuristics cannot guarantee aesthetic perfection on 100% of arbitrary footage.

---

## Quick Start (Zero-API Offline Editing)

### Windows (PowerShell)

```powershell
cd MotionLab-AI-ML01

# 1. One-time online bootstrap (installs pinned .venv + optional local Node renderers; ~450 MB disk)
powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\bootstrap.ps1

# 2. Verify local engines, resources, and licenses offline
.\.venv\Scripts\python.exe -m motionlab doctor

# 3. Copy your raw MP4 into input\raw.mp4 (original stays untouched) and edit offline in ONE command
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --offline --output "runs"

# 4. Inspect outputs in runs\<run_id>\ (final.mp4, contact_sheet.jpg, validation.json, README.md)
```

### Linux / macOS

```bash
cd MotionLab-AI-ML01

# 1. One-time online bootstrap
bash scripts/bootstrap.sh

# 2. Verify local environment offline
.venv/bin/python -m motionlab doctor

# 3. Run one-command offline edit
.venv/bin/python -m motionlab edit --input "input/raw.mp4" --profile auto --offline --output "runs"
```

---

## CLI Commands

```powershell
# System, engine & license diagnostics
.\.venv\Scripts\python.exe -m motionlab doctor [--json]

# List all 40 registered effects or render the 40-effect verification gallery
.\.venv\Scripts\python.exe -m motionlab effects list
.\.venv\Scripts\python.exe -m motionlab effects gallery --output "runs"

# Signal-only local analysis (cached by source SHA256 + config hash)
.\.venv\Scripts\python.exe -m motionlab analyze --input "input\raw.mp4" --output "runs"

# Build deterministic EditPlan JSON + explanation without rendering
.\.venv\Scripts\python.exe -m motionlab plan --input "input\raw.mp4" --profile auto --seed 42 --output "runs"

# Full offline edit (source aspect preserved by default)
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --seed 42 --offline --output "runs"

# Explicit 9:16 vertical preview + talking-head profile
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile talking-head --aspect 9:16 --quality preview --offline --output "runs"

# Strict neutral mode (fails closed if any required engine/check is missing)
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile neutral --offline --strict --output "runs"

# Also generate a fresh 9:16 preview alongside full-res final.mp4
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --preview-9x16 --offline --output "runs"

# Commercial orgs (>=4 people) without a Remotion Company License:
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --disable-remotion --offline --output "runs"

# Replay a saved edit_plan.json or re-validate an existing run folder
.\.venv\Scripts\python.exe -m motionlab render --plan "runs\<run_id>\edit_plan.json" --input "input\raw.mp4" --output "runs"
.\.venv\Scripts\python.exe -m motionlab validate --run "runs\<run_id>"
```

---

## Pipeline Architecture

```text
raw.mp4
  │
  ▼
Preflight (ffprobe JSON, rotation/SAR/VFR/HDR/audio probe, streaming SHA-256)
  │
  ▼
ResourceBudget (CPU/RAM/disk reserve preflight, bounded worker caps, timeout cleanup)
  │
  ▼
Local Async Analyzer (streaming downscaled cuts + optical flow + PCM RMS/onsets + optional OpenCV Haar face)
  │
  ▼
Candidate Events -> Deterministic Rule-Based Planner (7 profiles, cooldowns, safe zones, overlap rules)
  │
  ▼
edit_plan.json + plan_explain.json (canonical JSON, same input/config/seed -> identical plan_hash)
  │
  ▼
Multi-Engine Local Renderer:
  ├── FFmpeg Streaming Pipeline (40/40 effects + alpha composite + hard-locked audio mux)
  ├── Remotion 4.0.547 (offline VP9 yuva420p alpha SVG overlays; license-governed, opt-outable)
  └── HyperFrames 0.9.101 + local GSAP 3.15.1 (offline HTML/CSS/GSAP overlays; Apache-2.0)
  │
  ▼
final.partial.mp4 -> Technical Validator (ffprobe, full decode -xerror, audio streamhash + PCM sync, contact sheet)
  │
  ▼ (atomic rename ONLY on TECHNICAL_PASS)
runs/<run_id>/final.mp4
```

---

## Profiles (`motionlab/presets/`)

- `auto`: Deterministic measurable-signal selector (screen-like UI -> `tutorial`; high-motion HUD -> `gameplay`; consistent single face + audio -> `talking-head`; uncertain/3D/cartoon/multi-subject -> `neutral`). Never claims semantic niche recognition.
- `neutral`, `talking-head`, `product`, `storytelling`, `gameplay`, `tutorial`, `cinematic`.

---

## Legacy Upstream Reference

Upstream `browser-use/video-use` scripts (`helpers/`, `SKILL.md`, `install.md`, `skills/manim-video/`) are preserved untouched for legacy reference and baseline regression testing (`tests/test_render_*.py`). They are **never** imported or invoked by `motionlab`.

See [`docs/UPSTREAM.md`](./docs/UPSTREAM.md), [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md), and [`docs/ML01_IMPLEMENTATION_REPORT.txt`](./docs/ML01_IMPLEMENTATION_REPORT.txt) for full provenance, licensing, and verification evidence.

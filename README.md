# MotionLab Studio — ML-01.R5.R1 (Zero-API Premium Art Direction, Semantic Sound & Repository-Reuse Engine)

**MotionLab Studio (`ML-01.R5.R1`)** is an independent, 100% local, zero-API automatic video effects and generative motion-design CLI built atop upstream [`browser-use/video-use`](https://github.com/browser-use/video-use) (SHA `43cfc566833548093455c9d4a46bd69913b7e552`, MIT License).

Feed a raw MP4 (with existing visuals and voice audio), run **one local offline CLI command**, and MotionLab performs local offline ASR (`faster-whisper`), verified integer-cent StoryGraph extraction, local LLM scene planning (`llama.cpp` + `Qwen2.5-3B-Instruct`), all-frame Reels safe-zone & face/eye protection (`art_direction.py`), Two-Pass Semantic Sound Direction (`semantic_sound.py`) synced to compiled animation markers, local portrait matting (`MODNet ONNX`), single-browser dual-engine rendering (**Remotion MIT imported components + HyperFrames Apache-2.0 imported modules + FFmpeg streaming compositor**), voice-priority CC0-1.0 Foley mixing, active-window ablation QA, and full technical validation — **100% offline with \$0 per-edit API cost**.

> **Scope, Licensing & Honesty Disclosures**
> - **Zero runtime cloud/paid APIs:** No OpenAI, Anthropic, ElevenLabs, Google, Gemini, or remote render API is invoked at runtime. All cloud API keys are ignored/stripped and zero remote CDN/font assets are fetched.
> - **Repository Reuse First:** Ships **20 verified imported motion mechanisms** across **5 verified MIT/Apache-2.0 upstream repositories** (`11` Remotion MIT + `9` HyperFrames Apache-2.0), recorded in [`third_party/motion_sources.lock.json`](./third_party/motion_sources.lock.json) and [`third_party/imported_primitive_manifest.json`](./third_party/imported_primitive_manifest.json).
> - **Audio & CC0-1.0 Foley contract:** The original audio track is preserved untouched (`original_audio_master.m4a`, bit-exact packets and SHA-256). In `--audio-mode mix`, voice stays at unity (`1.0`, zero EQ/NR/pitch/tempo/automation) and music is OFF by default. Run `python scripts/provision_cc0_foley.py` once during bootstrap to download the curated 29-file 48 kHz 24-bit **CC0-1.0** SFX/Foley pack from `wyzowl/sfx-kit` (WAV binaries stay gitignored; exact SHA-256 checksums live in `assets/sfx_foley/manifest.json`). If local WAVs are absent, MotionLab honestly falls back to `PARTIAL_FOLEY_ASSETS` (`procedural_dsp_synthesis`).
> - **Technical vs Aesthetic gates:** Automated validation certifies technical integrity (`TECHNICAL_PASS`), while visual/audio taste remains human review (`PENDING_HUMAN_REVIEW`, VQ-01 rubric). No local VLM is installed (`VISION_CRITIC_NOT_RUN`), automated playback listening is not a human listener (`LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY`), and synthetic corpus tests do not replace real held-out footage (`REAL_HELDOUT_NOT_RUN`).
> - **Renderer licenses:** FFmpeg (`LGPL/GPL` external binary), HyperFrames (`Apache-2.0`), GSAP (`Standard No-Charge License`), and Remotion (`Remotion License` — free for individuals and organizations up to 3 people; organizations with >=4 people require a Remotion Company License or `--disable-remotion`). See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

---

## Quick Start (Windows PowerShell)

```powershell
cd MotionLab-AI-ML01

# 1. One-time online bootstrap (installs pinned .venv, local Node renderers, local AI models & CC0-1.0 Foley WAVs)
powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe scripts\provision_cc0_foley.py

# 2. Verify local engines, 20 imported primitives, CC0 foley, models, resources, and licenses offline
.\.venv\Scripts\python.exe -m motionlab doctor

# 3. Canonical ML-01.R5.R1 Generative Edit (100% offline, dual-engine Remotion + HyperFrames, voice+SFX mix, 9:16 preview)
.\.venv\Scripts\python.exe -m motionlab edit `
  --input "input\raw.mp4" `
  --profile auto `
  --offline `
  --composition generative `
  --enable-remotion `
  --audio-mode mix `
  --preview-9x16 `
  --seed 42 `
  --max-workers 2 `
  --output "runs"

# 4. Inspect outputs in runs\<run_id>\:
#    final.mp4, preview_9x16.mp4, contact_sheet.jpg, layout.json, continuity.json,
#    evidence\scene_peaks_phone_scale.jpg, evidence\sound_rationale.json,
#    evidence\frame_protection.csv, evidence\audio_event_qa.json, validation.json, README.md
```

### Linux / macOS

```bash
cd MotionLab-AI-ML01
bash scripts/bootstrap.sh
.venv/bin/python -m motionlab doctor
.venv/bin/python -m motionlab edit \
  --input "input/raw.mp4" \
  --profile auto \
  --offline \
  --composition generative \
  --enable-remotion \
  --audio-mode mix \
  --preview-9x16 \
  --seed 42 \
  --max-workers 2 \
  --output "runs"
```

---

## CLI Commands & Modes

```powershell
# Diagnostics (hardware, FFmpeg, Remotion, HyperFrames, 20 imported primitives, local models, licenses)
.\.venv\Scripts\python.exe -m motionlab doctor [--json]

# Legacy rule-based 40-effect list and gallery (fully preserved for backward compatibility)
.\.venv\Scripts\python.exe -m motionlab effects list
.\.venv\Scripts\python.exe -m motionlab effects gallery --output "runs"

# Signal-only local analysis (cached by source SHA-256 + config hash)
.\.venv\Scripts\python.exe -m motionlab analyze --input "input\raw.mp4" --output "runs"

# Build deterministic EditPlan without rendering
.\.venv\Scripts\python.exe -m motionlab plan --input "input\raw.mp4" --profile auto --seed 42 --output "runs"

# Full ML-01.R5 Generative Dual-Engine Edit (default R5 mode)
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --offline --composition generative --enable-remotion --audio-mode mix --preview-9x16 --seed 42 --max-workers 2 --output "runs"

# Preserve original audio track bit-exact with zero SFX (R3/R4 preserve-exact compatibility)
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --offline --composition generative --audio-mode preserve-exact --output "runs"

# Commercial orgs (>=4 people) without a Remotion Company License (HyperFrames + FFmpeg only)
.\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --offline --composition generative --disable-remotion --output "runs"

# Replay a saved edit_plan.json or re-validate an existing run folder
.\.venv\Scripts\python.exe -m motionlab render --plan "runs\<run_id>\edit_plan.json" --input "input\raw.mp4" --output "runs"
.\.venv\Scripts\python.exe -m motionlab validate --run "runs\<run_id>"
```

---

## ML-01.R5 Architecture

```text
raw.mp4 (immutable input, streaming SHA-256 verified before & after)
  │
  ▼
Preflight & ResourceBudget (ffprobe, disk reserve, bounded workers, scoped tree watchdog)
  │
  ├──► Offline Local Signal Analyzer (downscaled cuts, optical flow, PCM RMS/onsets, face/hand ROI)
  ├──► Offline ASR (faster-whisper-small, word-level timestamps, local_files_only=True)
  └──► Verified StoryGraph & Integer-Cent Ledger (conservation, negation/non-money isolation, anti-hallucination)
  │
  ▼
Local Generative Director (llama.cpp + Qwen2.5-3B-Instruct GGUF, dynamic enum JSON grammar)
  │  └─► Strictly bounded to Unified Primitive Registry (20 verified imported mechanisms)
  │      + Safe Generative Compiler (schema, NaN/depth/XSS/path guards, frame mapping, ablation plan)
  ▼
Dual-Engine Local Renderer + Local Matting (MODNet ONNX):
  ├── Remotion 4.0.547 + 11 Imported MIT Mechanisms (VP9 yuva420p alpha + behind-subject / AspectMorph / 3D FlipPage)
  ├── HyperFrames 0.9.101 + GSAP 3.15.1 + 9 Imported Apache-2.0 Mechanisms (data bars, counters, strikethrough, callouts, masks)
  └── FFmpeg Streaming Compositor (depth sandwich, macro cutout crop, safe-margin layout, H.264 yuv420p faststart)
  │
  ▼
Voice-Priority Audio Mixer (original_audio_master.m4a bit-exact + local voice-RMS calibrated SFX stem + matched no-SFX ref)
  │
  ▼
Technical Validator & Creative Proxy QA (720/720 decode -xerror, reuse gate, alpha ablation, audio QA, 360px phone sheets)
  │  (atomic rename final.partial.mp4 -> final.mp4 ONLY on TECHNICAL_PASS)
  ▼
runs/<run_id>/final.mp4 + preview_9x16.mp4 (direct downscale from final) + complete evidence artifacts
```

---

## Unified Imported Primitive Registry (`20` Verified Mechanisms)

| Engine | Upstream Repository | License | Verified Imported Mechanism IDs |
| --- | --- | --- | --- |
| Remotion | `riaz37/remotion-ui` | MIT | `rem_ru_counter`, `rem_ru_waterfall`, `rem_ru_strikethrough`, `rem_ru_masked_slide`, `rem_ru_track_matte`, `rem_ru_parallax` |
| Remotion | `vivmagarwal/remotion-effects-library` | MIT | `rem_viv_aspect_morph`, `rem_viv_particle_field` |
| Remotion | `RenderComp/free-remotion-templates` | MIT | `rem_rc_flip_page` |
| Remotion | `GTKottman/remotion-transitions` | MIT | `rem_gtk_frame_morph`, `rem_gtk_line_led` |
| HyperFrames | `heygen-com/hyperframes` (`0.9.101`) | Apache-2.0 | `hf_apple_money_count`, `hf_mk_progress_stat`, `hf_mk_specs_list`, `hf_mk_callout_highlight`, `hf_lt_mask_reveal`, `hf_strikethrough_replace`, `hf_organic_light_leak`, `hf_data_chart`, `hf_exploded_stack` |

Full provenance, upstream commit SHAs, file paths, adaptation notes, and Visual Benchmark Lab (`B01`–`B08`, `H1`–`H4`) results are documented in:
- [`docs/ML01_R5R1_BASELINE.txt`](./docs/ML01_R5R1_BASELINE.txt) & [`defects.json`](./defects.json)
- [`docs/ML01_R5R1_ADR.md`](./docs/ML01_R5R1_ADR.md)
- [`docs/ML01_R5R1_IMPLEMENTATION_REPORT.txt`](./docs/ML01_R5R1_IMPLEMENTATION_REPORT.txt)
- [`docs/ML01_R5_LINEAGE_RECONCILIATION.md`](./docs/ML01_R5_LINEAGE_RECONCILIATION.md)
- [`docs/ML01_R5_REPO_SURVEY.md`](./docs/ML01_R5_REPO_SURVEY.md)
- [`docs/ML01_R5_REUSE_REPORT.txt`](./docs/ML01_R5_REUSE_REPORT.txt)
- [`docs/ML01_R5_VISUAL_BENCHMARK.md`](./docs/ML01_R5_VISUAL_BENCHMARK.md)
- [`docs/ML01_R5_ADR.md`](./docs/ML01_R5_ADR.md)
- [`docs/ML01_R5_IMPLEMENTATION_REPORT.txt`](./docs/ML01_R5_IMPLEMENTATION_REPORT.txt)
- [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)
- Redacted `<=2 MiB` QA-01 evidence packs: [`qa_evidence/ML-01.R5.R1/`](./qa_evidence/ML-01.R5.R1/)

---

## Testing & Verification

```powershell
# Full unique pytest suite (unit + integration + negative + corpus + R1-R5 regressions)
.\.venv\Scripts\python.exe -m pytest -v tests/

# Remotion TypeScript type-check (covers all 11 imported MIT components + generative stage)
Push-Location renderers\remotion; npx tsc --noEmit; Pop-Location

# Legacy upstream browser-use/video-use tests (preserved separately, never double-counted)
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_render_*.py" -v
```

---

## Profiles (`motionlab/presets/`)

- `auto`: Deterministic measurable-signal selector (screen-like UI -> `tutorial`; high-motion HUD -> `gameplay`; consistent single face + audio -> `talking-head`; uncertain/3D/cartoon/multi-subject -> `neutral`).
- `neutral`, `talking-head`, `product`, `storytelling`, `gameplay`, `tutorial`, `cinematic`.

---

## Legacy Upstream Reference

Upstream `browser-use/video-use` files (`helpers/`, `SKILL.md`, `install.md`, `skills/manim-video/`) are preserved untouched for legacy reference and baseline regression testing (`tests/test_render_*.py`). They are **never** imported or invoked by `motionlab`.

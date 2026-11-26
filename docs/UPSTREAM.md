# Upstream Audit & Lineage (`browser-use/video-use`)

- **Upstream Repository**: `https://github.com/browser-use/video-use.git`
- **Upstream Remote Name**: `upstream` (read-only by convention; never push to `upstream`)
- **Cloned Upstream SHA**: `43cfc566833548093455c9d4a46bd69913b7e552`
- **Upstream Commit Date**: `2026-10-08T15:16:59-07:00` (`README: point cloud link to video-use.com waitlist (#222)`)
- **Clone / Audit Timestamp (UTC)**: `2026-11-26T10:52:54Z`
- **Local Development Branch**: `feat/ml-01-zero-api-effects`
- **Upstream License**: MIT (`LICENSE`, Copyright (c) 2026 Browser Use — preserved intact)

## Audited Upstream File Map

| Upstream Path | Audited Role | ML-01 Disposition |
|---|---|---|
| `LICENSE` | MIT License (Copyright 2026 Browser Use) | Preserved intact |
| `README.md` | Upstream agent/Claude Code + ElevenLabs overview | Updated to document ML-01 local-first CLI while referencing legacy workflow |
| `install.md` | Upstream interactive install (requires ElevenLabs API key + agent symlinks) | Preserved as legacy reference; **NOT** executed by ML-01 |
| `SKILL.md` | Upstream conversation-driven LLM editing prompt | Preserved as legacy reference; **NOT** executed by ML-01 |
| `.env.example` | Contains `ELEVENLABS_API_KEY=` | Updated to note local-first zero-API default; legacy key strictly opt-in |
| `pyproject.toml` | Upstream deps (`requests`, `librosa`, `matplotlib`, `pillow`, `numpy`) | Extended with `motionlab` package & verified local-only core deps; `requests` kept legacy-only |
| `helpers/transcribe.py` | Calls ElevenLabs Scribe (`https://api.elevenlabs.io/v1/speech-to-text`) | Strictly isolated legacy module; **NEVER** imported by `motionlab` |
| `helpers/transcribe_batch.py` | Parallel batch wrapper for `transcribe.py` | Strictly isolated legacy module; **NEVER** imported by `motionlab` |
| `helpers/pack_transcripts.py` | Packs Scribe JSON into `takes_packed.md` | Preserved as legacy reference |
| `helpers/render.py` | EDL cut renderer (`extract_segment`, `concat_segments`, `build_master_srt`, `apply_loudnorm_two_pass`) | Preserved untouched for upstream baseline tests; not used directly by ML-01 because it cuts timelines, adds 30ms audio fades, re-encodes audio, and applies loudnorm by default |
| `helpers/grade.py` | FFmpeg `signalstats` auto-grade & presets | Preserved untouched as legacy reference |
| `helpers/timeline_view.py` | Filmstrip + waveform composite visualizer | Preserved untouched as legacy reference |
| `skills/manim-video/` | Manim CE skill documentation & helper scripts | Preserved untouched as optional reference |
| `tests/test_render_captions.py` | 10 unit tests for `helpers/render.py` captions | Preserved and executed in baseline & regression |
| `tests/test_render_orientation.py` | 6 unit tests for `helpers/render.py` orientation | Preserved and executed in baseline & regression |
| `tests/test_render_fps.py` | 10 unit tests for `helpers/render.py` FPS parsing/probing | Preserved and executed in baseline & regression |
| `poster.html`, `static/` | Upstream visual assets | Preserved untouched |

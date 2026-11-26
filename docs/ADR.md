# Architecture Decision Records (ADR) — MotionLab-AI ML-01

## ADR-001: Isolated `motionlab/` Package Alongside Legacy `helpers/`
- **Context**: Upstream `browser-use/video-use` is a Claude Code skill with ElevenLabs Scribe cloud transcription (`helpers/transcribe.py`) and cut-based EDL rendering (`helpers/render.py`) that applies 30ms audio fades, re-encodes audio, and defaults to `-14 LUFS` loudnorm.
- **Decision**: Keep all upstream `helpers/`, `SKILL.md`, `install.md`, and `tests/test_render_*.py` intact for legacy reference and baseline regression. Build ML-01 inside a completely separate `motionlab/` Python package that never imports `helpers/transcribe*`, `requests`, or any network client.
- **Consequences**: Zero API calls in runtime path; 26/26 upstream baseline tests stay untouched and passing.

## ADR-002: Integer Frame Indexing & Non-Destructive Full-Length Audio-Master Timeline
- **Context**: Float timestamp accumulation causes A/V drift and off-by-one frame drops; upstream `render.py` cuts footage and re-encodes audio.
- **Decision**: Represent all effect spans, scene cuts, and timeline boundaries using integer frame indices `[start_frame, end_frame_exclusive)` paired with an exact rational `Fraction(fps_num, fps_den)` timebase. Render video frames independently (`video_only.mp4`) and mux the original source audio stream(s) once at the end with `-c:a copy` whenever container-compatible (or disclosed high-bitrate AAC with PCM correlation check when source audio codec is incompatible with MP4, e.g. raw PCM). Never use `-shortest`.
- **Consequences**: Exact frame count and source audio preservation without fades, retime, loudnorm, or pitch shifts.

## ADR-003: Streaming Frame-Exact Procedural Compositor + Multi-Engine Architecture
- **Context**: Implementing 40 distinct eased effects (curved SDF masks, overscan-calculated micro-tilt, damped springs, duration-preserving boundary transitions, audio-reactive waveforms) purely via FFmpeg filtergraph strings causes `zoompan` jitter, extreme `geq` slowness, and expression bugs. Conversely, forcing headless Chrome for every video frame is too slow on consumer laptops.
- **Decision**:
  1. Primary local engine (`ffmpeg`): streams decoded raw frames from FFmpeg via bounded pipes (no full-video RAM buffering; small sliding ring buffer of <=16 frames for transitions/echo), applies vectorized NumPy/OpenCV/Pillow frame-exact transforms/masks/color/overlays evaluated strictly from integer frame index `f`, and pipes encoded frames directly into FFmpeg `libx264` (`yuv420p`) followed by single lossless audio master mux.
  2. Secondary engine (`hyperframes`, Apache-2.0): real locked project in `renderers/hyperframes/` using local pinned `hyperframes` CLI + GSAP + local Chrome executable (zero runtime downloads) for verified HTML/CSS/GSAP graphic overlay templates with alpha compositing onto the timeline.
  3. Secondary engine (`remotion`, license-gated): real locked React/TS project in `renderers/remotion/` using local pinned `remotion` CLI + local Chrome executable (zero runtime downloads) for verified React/SVG overlay compositions, explicitly gated behind `remotion_license_eligible` compliance policy.
  4. Optional engine (`manim`): stubbed/gated adapter not on the critical path.
- **Consequences**: All 40 effects render cleanly offline at high speed with deterministic frame-exact math, while genuine HyperFrames and Remotion pipelines are also available, tested, and license-compliant.

## ADR-004: Disk Reserve Governance on Constrained Host Volumes
- **Context**: Phase F suggests reserving `max(5 GiB, 10% volume capacity)` configurable based on system. On typical developer laptops (e.g., 460 GiB volume with 27 GiB currently free), 10% equals 46 GiB > free space, which would erroneously block every run if hardcoded without configurability.
- **Decision**: Default `--min-free-gb` to `min(max(5.0, 0.10 * total_gb), 5.0)` when unconfigured — i.e., `5.0 GiB` safe floor by default — while exposing `--min-free-gb`, `--max-temp-gb`, and `--reserve-pct` explicitly, logging the volume free-space warning whenever free space is below `10%` of volume capacity, and strictly aborting if `free_bytes - estimated_temp_bytes < min_free_bytes`.
- **Consequences**: Safe disk protection that prevents filling the user's disk without falsely blocking healthy local runs on 90%-utilized developer SSDs.

## ADR-005: Pure NumPy STFT Audio Analysis & Bundled OpenCV Haar Tracking
- **Context**: Adding heavy ML libraries (PyTorch, MediaPipe, librosa+numba+llvmlite) bloats install size, introduces Python 3.13 wheel fragility, or triggers runtime model downloads.
- **Decision**: Extract temporary 16 kHz mono PCM via FFmpeg in bounded chunks; compute frame-aligned RMS/peak envelopes, silence windows, clipping flags, and STFT spectral-flux onsets deterministically in pure NumPy. Use bundled Apache-2.0/Intel OpenCV Haar cascades (`haarcascade_frontalface_default.xml`, `haarcascade_frontalface_alt2.xml`) with temporal IoU/EMA smoothing and explicit low-confidence fallback for optional subject framing.
- **Consequences**: Fast, lean (<70 MB `.venv`), 100% offline, deterministic, cross-platform Windows/macOS/Linux.

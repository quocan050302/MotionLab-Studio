# Third-Party Notices & License Verification

This file documents the verified licenses, versions, and usage conditions of all upstream code, Python dependencies, Node.js renderer engines, binary tools, and built-in model assets used by **MotionLab-AI ML-01**.

## 1. Upstream Base Repository
- **browser-use/video-use** (SHA `43cfc566833548093455c9d4a46bd69913b7e552`)
  - **License**: MIT License (Copyright (c) 2026 Browser Use)
  - **File**: `LICENSE` (preserved verbatim at repository root)

## 2. Python Runtime & Test Dependencies (`.venv`, verified `pip freeze`)
- **NumPy** (`2.2.6`) — BSD-3-Clause License
- **Pillow** (`13.0.0`) — MIT-CMU License
- **opencv-python-headless** (`4.14.0.94`) — Apache-2.0 License
  - Bundled Haar cascades (`haarcascade_frontalface_default.xml`, `haarcascade_frontalface_alt2.xml`, `haarcascade_profileface.xml`): Intel License Agreement / BSD-compatible OpenCV distribution license. Bundled inside the installed wheel; zero network download at edit runtime.
- **psutil** (`7.2.2`) — BSD-3-Clause License
- **pytest** (`9.2.0`), **pluggy** (`1.7.0`), **iniconfig** (`2.3.1`), **packaging** (`26.4`), **Pygments** (`2.21.0`) — MIT / Apache-2.0 / BSD-2-Clause (dev/test tooling)

## 3. Media Toolchains (Local Binaries)
- **FFmpeg** (verified local build `6.0`) & **ffprobe** (verified local build `4.4-tessus`)
  - **License**: LGPL v2.1+ or GPL v2+/v3 depending on build flags.
  - **Important Notice**: The verified local test build on this macOS host was compiled with `--enable-gpl --enable-version3 --enable-nonfree` and includes `libx264` (GPL v2) and `libvpx` (BSD-3-Clause). If distributing bundled binaries commercially, users/distributors must ensure compliance with FFmpeg/x264 GPL or LGPL licensing (e.g., invoke user-installed system FFmpeg or use LGPL-compliant codecs/encoders such as hardware VideoToolbox/MediaFoundation/OpenH264). MotionLab-AI invokes external `ffmpeg`/`ffprobe` via subprocess and does not bundle FFmpeg binaries in git.

## 4. Optional Node.js Renderer Engines

### HyperFrames (`renderers/hyperframes/`)
- **hyperframes** (`0.9.98`) by HeyGen — **Apache-2.0 License**
  - Open-source browser-native HTML/CSS/JS deterministic frame renderer using local Chrome/Chromium and FFmpeg.
- **GSAP** (`3.15.1`, GreenSock Animation Platform) — **Standard "No Charge" GreenSock License** (`https://gsap.com/standard-license`)
  - Free for personal and commercial use, **with the restriction** that GSAP may not be used to build or help create a visual no-code website/animation editor that competes with Webflow. MotionLab-AI ML-01 is a command-line rule-based automatic video effects engine (not a visual/no-code editor competing with Webflow). HyperFrames templates can also run with pure deterministic CSS/JS seekable timelines without GSAP if an organization prefers Apache-2.0/MIT-only stacks.

### Remotion (`renderers/remotion/`)
- **remotion** (`4.0.547`) — **Remotion Source-Available License** (NOT MIT; verified at `https://www.remotion.dev/license`)
  - **Free Tier**: Free for individuals, non-profits, evaluators, and for-profit organizations with **up to 3 people**.
  - **Paid Company License Required**: For-profit organizations with **4 or more people** MUST purchase a Remotion Company License (e.g., Remotion for Creators or Remotion for Automators for automated video pipelines).
  - **ML-01 Policy Guard**: MotionLab-AI does **NOT** bypass Remotion's license. The Remotion renderer is explicitly license-gated (`--enable-remotion` / `remotion_license_eligible=true` after confirming eligibility). When disabled or ineligible, `doctor` marks Remotion as `disabled (license-gated)` or `unavailable` and ML-01 runs cleanly via the FFmpeg and HyperFrames engines.

## 5. Optional / Out-of-Critical-Path Engines
- **Manim Community Edition** — MIT License (optional, not installed by default at ML-01; `motionlab/renderers/manim.py` reports `unavailable` unless explicitly installed and invoked with explicit user data/template).

## 6. Fonts & Assets
- ML-01 renders **zero subtitles and zero decorative text filler by default**.
- Contact sheets and QA diagnostic labels use Pillow built-in bitmap font fallback or read-only OS system fonts (e.g., Menlo/DejaVuSansMono/Consolas) only for developer QA artifacts (`contact_sheet.jpg`), never embedded into user deliverables.
- Test fixtures in `tests/fixtures/` are procedurally generated via Python/FFmpeg (100% synthetic original test patterns); no third-party copyrighted stock video, audio, music, or voice assets are bundled or downloaded.

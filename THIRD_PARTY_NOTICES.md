# Third-Party Notices & License Compliance — MotionLab AI (ML-01)

MotionLab AI (ML-01) preserves upstream copyright and license obligations and documents the actual licenses of every runtime engine, dependency, model weight, and asset used locally.

## 1. Upstream Repository
- **Project:** `browser-use/video-use`
- **Upstream Commit SHA:** `43cfc566833548093455c9d4a46bd69913b7e552`
- **License:** MIT License (`Copyright (c) 2026 Browser Use`) — full text preserved verbatim in `LICENSE`.

## 2. Python Runtime Dependencies (`.venv` / `requirements.lock`)
| Package | Pinned Version | License | Role in ML-01 |
|---|---|---|---|
| `numpy` | `2.5.5` | BSD-3-Clause | Deterministic array math, audio RMS/FFT spectral flux, frame transforms |
| `pillow` | `13.0.0` | MIT-CMU | Anti-aliased RGBA overlays, masks, contact sheets, gallery rendering |
| `opencv-python-headless` | `4.14.0.94` | Apache-2.0 | Optical flow, HSV histogram cut detection, sub-pixel affine warps, bundled Haar face cascade |
| `scipy` | `1.18.1` | BSD-3-Clause | Signal/numeric utilities |
| `psutil` | `7.2.2` | BSD-3-Clause | CPU, RAM, and owned child process-tree governance |
| `pytest` (dev/test only) | `9.2.0` | MIT | Automated unit, integration, E2E, and regression test execution |

### Bundled Local Model Weights
- **OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)** — bundled inside `opencv-python-headless==4.14.0.94` (`cv2.data.haarcascades`).
- **License:** Apache-2.0 (Intel / OpenCV). Zero runtime downloads; zero biometric identity recognition.

## 3. System Binary Codecs & Media Engines
- **FFmpeg / ffprobe** (tested on `9.0.2-full_build-www.gyan.dev` with `libx264`, `libvpx-vp9`, `aac`):
  - **License:** GPLv3 (for `--enable-gpl --enable-version3` builds including `libx264`) or LGPLv2.1+ (for LGPL builds without GPL encoders). Users distributing binaries or SaaS services must comply with their installed FFmpeg build's license terms.

## 4. Optional Local Node Renderers (`renderers/`)

### Remotion (`renderers/remotion/` — `remotion@4.0.547`, `@remotion/cli@4.0.547`)
- **License:** Remotion License (custom dual license — **NOT MIT**). Official terms: <https://www.remotion.dev/license>
  - **Free License:** Free for commercial and non-commercial use by **individuals**, **non-profit organizations**, **evaluators**, and **for-profit organizations with up to 3 people** (total headcount).
  - **Company License (Paid):** Required for **for-profit organizations with 4 or more people** (Creators or Automators plan).
- **Governance in ML-01:** MotionLab AI never bypasses Remotion's license. Organizations with ≥4 people without a Remotion Company License MUST pass `--disable-remotion` or set `MOTIONLAB_ALLOW_REMOTION=0`; ML-01 automatically marks Remotion unavailable and uses certified FFmpeg/HyperFrames fallback.
- **Bundled React:** `react@19.1.0`, `react-dom@19.1.0` (MIT License).

### HyperFrames (`renderers/hyperframes/` — `hyperframes@0.9.101`) & GSAP (`gsap@3.15.1`)
- **HyperFrames License:** Apache-2.0 License (<https://github.com/heygen-com/hyperframes>).
- **GSAP License:** Standard "No Charge" License (<https://gsap.com/standard-license>). Free for general websites, web apps, commercial client projects, and code-based animation tools; prohibits embedding GSAP inside competing no-code/visual animation builders that directly rival Webflow's visual canvas.
- **Offline Provisioning:** `gsap.min.js` is vendored locally from `renderers/hyperframes/node_modules/gsap/dist/gsap.min.js` into ephemeral render slots; zero CDN fetches occur at runtime.

### Local Browser (Headless Rendering Host)
- Remotion and HyperFrames invoke the user's already-installed local Chrome/Edge/Chromium browser (e.g., Google Chrome / Microsoft Edge) offline without downloading Chromium during editing.

## 5. Optional / Disabled Components
- **Manim Community Edition** (`motionlab/renderers/manim.py`): MIT License; gated off the ML-01 critical path and disabled by default.
- **Legacy Cloud Helpers (`helpers/transcribe.py`, `helpers/transcribe_batch.py`)**: Upstream MIT code calling ElevenLabs API. Isolated and **NEVER** imported or executed by MotionLab AI ML-01 runtime.
- **Fonts & Assets:** ML-01 adds NO default subtitles, NO remote web fonts, NO stock footage, NO default music or SFX, and NO third-party copyrighted media. All test fixtures are generated procedurally via FFmpeg/NumPy or supplied locally by the owner.

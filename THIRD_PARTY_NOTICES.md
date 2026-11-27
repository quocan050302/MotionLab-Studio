# Third-Party Notices & License Compliance — MotionLab Studio (ML-01.R5)

MotionLab Studio preserves all upstream copyright notices, license files, and attribution obligations across core dependencies, imported motion primitives, local AI models, and rendering engines.

## 1. Upstream Repository
- **Project:** `browser-use/video-use`
- **Upstream Commit SHA:** `43cfc566833548093455c9d4a46bd69913b7e552`
- **License:** MIT License (`Copyright (c) 2026 Browser Use`) — verbatim text preserved in `LICENSE`.

## 2. Imported Upstream Motion Primitives (ML-01.R5 — `third_party/motion_sources.lock.json`)
Every imported folder contains its verbatim upstream `LICENSE` and `NOTICE` file. Demo media, trademarks, remote fonts, and sample audio were stripped prior to import.

| Origin Slug | Repository | Pinned Commit SHA | License | Copyright / Attribution | Imported Mechanisms |
|---|---|---|---|---|---|
| `heygen-hyperframes` | `https://github.com/heygen-com/hyperframes` | `4fca68b23561e3f1271b876ed4715dce9c7cf208` | **Apache-2.0** | HeyGen / HyperFrames contributors (`LICENSE` SHA-256 `4259155fb06f`) | 9 blocks (`apple-money-count`, `mk-progress-stat`, `mk-specs-list`, `mk-callout-highlight`, `lt-mask-reveal`, `strikethrough-replace`, `organic-light-leak-overlay`, `data-chart`, `exploded-stack`) |
| `remotion-ui` | `https://github.com/riaz37/remotion-ui` | `27e51ae809eab7c8f85933ad3725aa6a6d3d77c3` | **MIT** | `Copyright (c) 2026 RemotionUI` (`LICENSE` SHA-256 `97d885d1d43e`) | 6 components (`Counter`, `WaterfallChart`, `StrikethroughReplace`, `MaskedSlideReveal`, `TrackMatte`, `ParallaxLayers`) + 6 helper libs |
| `remotion-effects-library` | `https://github.com/vivmagarwal/remotion-effects-library` | `f2781731e10e5c764498704bbbac7ca9466f5866` | **MIT** | `Copyright (c) 2026 Vivek Agarwal` (`LICENSE` SHA-256 `941b57e5a610`) | 2 components (`AspectMorphCard`, `ParticleField`) |
| `free-remotion-templates` | `https://github.com/RenderComp/free-remotion-templates` | `f648980e528defb7594e24c6447055f0540f340b` | **MIT** | `Copyright (c) 2026 Trimora Inc.` (`LICENSE` SHA-256 `3e0f0f2895a6`) | 1 component (`FlipPageTransition`) |
| `remotion-transitions` | `https://github.com/GTKottman/remotion-transitions` | `58df89999661dacef23bc23d05b9d2cd978bda45` | **MIT** | `Copyright (c) 2026 GTKottman and contributors` (`LICENSE` SHA-256 `c66a3f29317e`) | 2 transitions (`FrameMorphTransition`, `LineLedTransition`) |

## 3. Local Offline AI Models & Inference Runtimes (`models/manifest.json`)
Model weights and native inference binaries are downloaded via one-time explicit bootstrap (`scripts/bootstrap_local_models.py`), verified by SHA-256, ignored by Git (`.gitignore`), and executed strictly offline (`local_files_only=True`).

| Component | Version / Source | License | Role |
|---|---|---|---|
| **Faster-Whisper Small** (`Systran/faster-whisper-small`) | CTranslate2 int8/fp32 | **MIT** | Local offline speech-to-text & exact word timestamps |
| **Qwen2.5-3B-Instruct** (`Qwen/Qwen2.5-3B-Instruct-GGUF`, Q4_K_M) | Alibaba Qwen | **Apache-2.0** | Local grammar-constrained JSON creative director |
| **llama.cpp** (`ggml-org/llama.cpp`, release `b12803`) | Windows x64 CPU | **MIT** | Local GGUF inference engine (`llama-completion.exe`) |
| **MODNet Portrait Matting** (`modnet_photographic_portrait_matting.onnx`) | ZHKKKe / MODNet | **Apache-2.0** | Local foreground subject segmentation via ONNX Runtime |
| **OpenCV Haar Cascade** (`haarcascade_frontalface_default.xml`) | Bundled in OpenCV 4.14 | **Apache-2.0** | Local face ROI safety protection (zero biometric ID) |

## 4. Python Runtime Dependencies (`.venv`)
| Package | License | Role |
|---|---|---|
| `numpy` | BSD-3-Clause | Deterministic array math, integer-cent/audio DSP, frame transforms |
| `pillow` | MIT-CMU | Contact sheets, masks, anti-aliased RGBA compositing |
| `opencv-python-headless` | Apache-2.0 | Optical flow, scene/cut analysis, ROI tracking, video streaming |
| `scipy` | BSD-3-Clause | Physical-modeling audio synthesis filters (`butter`, `sosfilt`) |
| `psutil` | BSD-3-Clause | Process-tree RSS and system resource governance |
| `faster-whisper` / `ctranslate2` | MIT | Local ASR execution |
| `onnxruntime` | MIT | Local CPU MODNet inference |
| `pytest` (dev/test) | MIT | Automated test verification |

## 5. System Codecs & Local Node Renderers (`renderers/`)
- **FFmpeg / ffprobe** (tested on `9.0.2-full_build`): GPLv3 (`--enable-gpl` / `libx264`) or LGPLv2.1+ depending on installed build.
- **Remotion (`remotion@4.0.547`, `@remotion/cli@4.0.547`):**
  - **License:** Remotion License (source-available, **NOT MIT**; see <https://www.remotion.dev/license>). Free for individuals, non-profits, evaluators, and for-profit organizations with up to 3 people; paid Company License required for for-profit organizations with 4+ people.
  - **Enforcement:** Gated behind explicit `--enable-remotion` opt-in / `MOTIONLAB_ALLOW_REMOTION`. Free MIT component code does NOT grant Remotion rendering rights.
  - **React / ReactDOM (`19.1.0`):** MIT License.
- **HyperFrames (`hyperframes@0.9.101`):** Apache-2.0 License (<https://github.com/heygen-com/hyperframes>).
- **GSAP (`gsap@3.15.1`):** Standard "No Charge" License (<https://gsap.com/standard-license>, **NOT MIT**). Vendored locally as `./gsap.min.js`; zero runtime CDN fetches.

## 6. Audio & Media Assets
- **Default SFX:** Local physical-modeling procedural DSP (`procedural_dsp_synthesis`, MIT). Honestly reported as `PARTIAL_FOLEY_ASSETS` unless owner-supplied verified CC0/commercial recorded WAVs are placed in `assets/sfx_foley/` with SHA-256 hashes in `assets/sfx_foley/manifest.json`.
- **Music & Subtitles:** Disabled by default (`--music off`, zero auto-captions).
- **Legacy Cloud Helpers (`helpers/transcribe.py`):** Upstream MIT ElevenLabs helpers remain isolated and are **NEVER** imported or executed by MotionLab Studio runtime.

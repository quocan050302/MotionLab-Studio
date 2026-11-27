# MotionLab Studio — ML-01.R5 Lineage Reconciliation Report (A0)

## 1. Git Ancestry & Preservation
- **Remote HEAD before A0 (`origin/feat/ml-01-zero-api-effects`):** `f374b7e23c03ecf9dfb18a101e73273f05beb3f9`
- **Preserved local recovery ref (Windows HEAD):** `refs/recovery/ml01_win_head_20261127` -> `f374b7e23c03ecf9dfb18a101e73273f05beb3f9`
- **Historical R4 HEAD (separate lineage):** `6bcdb94e2a092274db1deb11d6d03dd9c4b81fe6` (fetched cleanly by exact SHA, mounted read-only at `.cache/r4_hist_readonly`)
- **Preserved local recovery ref (Historical R4):** `refs/recovery/ml01_r4_hist_20261127` -> `6bcdb94e2a092274db1deb11d6d03dd9c4b81fe6`
- **Shared Upstream Merge Base (`browser-use/video-use` MIT):** `43cfc566833548093455c9d4a46bd69913b7e552`
- **Local Staging Branch:** `integration/ml01-r5-20261127` descending cleanly from `f374b7e23c03ecf9dfb18a101e73273f05beb3f9` (zero `reset --hard`, zero `force push`, zero writes to `master` or `upstream`).

## 2. Primary Input Verification (Read-Only, Immutable)
- **File:** `input/Olivia_Paycheck_V5_Kitchen_30s_Combined (1).mp4`
- **SHA-256:** `c56c9ae0b6273a32c3847d4eae9d9015a5b1c3f911776435071a496dbb44ee7e`
- **Streams:** H.264 `720x1280`, `24/1` FPS, `720` frames, `30.000s`; AAC mono `44100 Hz`, `1296` packets, `start_pts=0`, `initial_padding=1024`.

## 3. Reconciled Architecture & Fixes Made During Port
1. **Preserved Windows Foundation (`f374b7e`):** Kept `EditConfig`, `MediaInfo`, `ResourceBudget`, 40 ML-01 signal effects, offline Remotion `4.0.547` + HyperFrames `0.9.101` adapters, and all 91 passing baseline tests intact. Added non-breaking `RationalFraction` (`isinstance(..., Fraction) == True`) and property bridges (`MediaInfo.path`, `MediaInfo.duration_frames`, `ResourceTracker`, `RuntimeResourceWatchdog`).
2. **Ported Proven R2–R4 Capabilities (`6bcdb94`):**
   - `motionlab/semantic/{asr,local_model,grounding,niche,director,schema}.py`
   - `motionlab/vision/{segmentation,prop_mask,tracking,anchors}.py`
   - `motionlab/audio/{schema,synthesis,cues,mixer}.py`
   - `motionlab/composition/{schema,recipes,vfx,planner,engine,r4_engine,quality}.py`
   - `motionlab/renderers/hyperframes_vfx.py` + full-frame methods in `remotion.py` / `hyperframes.py`
   - `renderers/remotion/src/CreativeSceneComposition.tsx` + `src/composition/{schema.ts,easing.ts,VfxSceneComposition.tsx}`
3. **Real Windows & FFmpeg 9.0.2 Fixes Caught During Live A0.3 Verification:**
   - **Windows CPU LLM Backend (`local_model.py`):** Replaced macOS-hardcoded `-ngl 99` and `DYLD_LIBRARY_PATH` with cross-platform backend selection (`-ngl 0` on Windows CPU with `llama.cpp b12803` MIT prebuilt `llama-completion.exe`, `-ngl 99` on macOS Apple Silicon).
   - **FFmpeg 9.0.2 `-vsync 0` Removal (`segmentation.py`, `hyperframes.py`):** Replaced removed `-vsync 0` option with `-fps_mode passthrough`.
   - **CompositionGraph URL Sanitization (`vfx.py`, `planner.py`):** Stripped human-readable license doc URLs from `engine_capabilities` stored inside `CompositionGraph` while keeping `_scan_forbidden_strings` strictly fail-closed.
   - **Fixed R4 `vfx-benefit-ledger` Index-Bar Bug (`hyperframes_vfx.py`):** Replaced fake `bar_w = int(card_w * (0.82 - idx * 0.18))` and tiny `26/14px` text with actual numeric proportion parsing and phone-readable typography.

## 4. A0.3 Live Windows Recovery Verification
- **HEAD Baseline (`f374b7e`):** `.\.venv\Scripts\python.exe -m pytest -q tests/` -> **91 passed, 15 subtests passed, 0 failed, 0 skipped** (74.05s).
- **Remotion TypeScript Check:** `npx tsc --noEmit` in `renderers/remotion/` -> **exit 0**.
- **Pinned Local Models Provisioned & SHA-256 Verified:**
  - `models/modnet_photographic_portrait_matting.onnx` (Apache-2.0, `07c308cf0fc7...`)
  - `models/faster-whisper-small/` (MIT, `model.bin` `3e305921506d...` + all 4 files verified)
  - `models/qwen2.5-3b-instruct-q4_k_m.gguf` (Apache-2.0, `626b4a6678b8...`)
  - `tools/llama.cpp/llama-completion.exe` (MIT, `ggml-org/llama.cpp` release `b12803`, archive SHA-256 `c9d69aef51a6...`)
- **Live Windows R4 Recovery Dry-Run (`runs/a0_recovery_check/20261127T030243Z_c56c9ae0_001`):**
  - Executed real offline Faster-Whisper ASR (`70 words`, `lang=en`, `prob=0.9982`, `9.16s`), local Qwen2.5-3B structured JSON director, MODNet ONNX portrait matting, GrabCut prop isolation, local SFX timeline + master-preserving mix, and 8-scene R4 `CompositionGraph` (`graph_hash=ba4c0ae61aa872e4bf061f4ccddcd64fad0c4f62d4aeb9b6f3b4d3600e5507ab`, `exit 0`).
- **Honest Status:** R4 aesthetic baseline remains owner-rejected (not promoted as R5). A0 reconciliation is **COMPLETE**; machine capability matrix recorded in `lineage_matrix.json`.


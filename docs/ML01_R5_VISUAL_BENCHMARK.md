# MotionLab Studio ML-01.R5 — Visual Benchmark Lab Report (B01–B08)

**Status:** `8/8 RENDERED & VISUALLY INSPECTED AT 360px PHONE SCALE` | **Human Gate (G8):** `PENDING_HUMAN_REVIEW`  
**Competitor Reference Status:** `REFERENCE_NOT_AVAILABLE` (internal visual benchmark only)  
**Foley Library Status:** `PARTIAL_FOLEY_ASSETS` (`procedural_dsp_synthesis`, zero mislabeled studio claims)

## 1. Benchmark Summary Table (Native `720×1280 @ 24fps`, H.264 + AAC 192k Mix)

| ID | Benchmark | Engines Executed | Imported Mechanisms Used | Frames | Motion MAD | SFX Peak / True Peak | Artifact SHA-256 (short) | Visual Inspection & Polish Applied |
|---|---|---|---|---|---|---|---|---|
| **B01** | Financial Flow (Proportional Ledger + Counter) | `remotion`, `hyperframes` | `rem_ru_waterfall`, `hf_apple_money_count` | 96 (4.0s) | 28.43 | `-5.65 dBFS` / `-5.7 dBTP` | `a36464cb7923` | Crisp ivory editorial waterfall (`$1,200→$100`); caught bill tokens overlapping `$100` -> fixed `.money-burst z-index: 1 < .stage-wrap z-index: 5`. |
| **B02** | Kinetic Typography & Contrast Hierarchy | `remotion`, `hyperframes` | `rem_ru_strikethrough`, `rem_ru_masked_slide`, `hf_mk_callout_highlight` | 84 (3.5s) | 43.52 | `-6.11 dBFS` / `-6.1 dBTP` | `6d19823a6152` | Razor-sharp hierarchy at 360px (`BALANCE ≠ SPENDABLE`, Georgia serif takeaway, word-sweep callout); zero overlap. |
| **B03** | Atmospheric Depth Particles Behind Subject Matte | `remotion` | `rem_viv_particle_field`, `rem_ru_counter`, `rem_ru_parallax` | 84 (3.5s) | 32.20 | `-7.29 dBFS` / `-3.4 dBTP` | `c1acd4a71cff` | Real MODNet hair/shoulder silhouette matte; caught head crown covering center digits at `y=10%` -> raised behind-subject counter to `y=4.5%` for A6 & B08! |
| **B04** | Conceptual Product Reveal (3D Page Turn + Specs) | `remotion`, `hyperframes` | `rem_rc_flip_page`, `hf_mk_specs_list` | 84 (3.5s) | 120.87 | `-5.84 dBFS` / `-6.0 dBTP` | `bddc37e2842b` | Genuine 3D `rotateY` page turn (`The Next Paycheck Plan` -> `56-page workbook`) + grounded specs; cleaned raw `ASSET_MISSING` tag out of viewer copy while keeping disclosure in metadata. |
| **B05** | Editorial Layout Morph + Proportional Data Chart | `remotion`, `hyperframes` | `rem_viv_aspect_morph`, `hf_data_chart` | 84 (3.5s) | 78.22 | `-7.69 dBFS` / `-3.6 dBTP` | `f053c11ae1ba` | Smooth layout-rect morph (`objectFit: cover`, zero squash) beneath exact proportional bars (`$800/$200/$100/$100 of $1,200`). |
| **B06** | Seamless Frame-Preserving Transition + Light Leak | `remotion`, `hyperframes` | `rem_gtk_line_led`, `rem_ru_masked_slide`, `hf_organic_light_leak` | 84 (3.5s) | 62.09 | `-6.35 dBFS` / `-1.2 dBTP` | `09d2c3b586e2` | Clean brass rail aperture split into warm amber-glow title stage; exact 84 frames, 0 black/dropped frames. |
| **B07** | Tracked Hands Macro Aperture + Proportional Stat | `remotion`, `hyperframes` | `rem_ru_track_matte`, `rem_ru_counter`, `hf_mk_progress_stat` | 84 (3.5s) | 37.97 | `-6.60 dBFS` / `-3.9 dBTP` | `ed34199357d9` | Proportional 67% `-$800 BILLS` bar + `$400` ribbon; caught macro box touching chin at `y=0.35` -> lowered to `y=0.50` & moved top bar to `y=54px` for A6! |
| **B08** | Spatial 2.5D Multi-Plane Parallax Behind Subject | `remotion`, `hyperframes` | `rem_ru_parallax`, `hf_apple_money_count` | 84 (3.5s) | 32.58 | `-6.94 dBFS` / `-2.2 dBTP` | `24f9a10a81b7` | Verified post-polish: `"BANK BALANCE $1,200"` 100% readable at `y=4.5%`, bill tokens cleanly behind plate (`z-index: 1`), Olivia MODNet foreground cleanly in front! |

## 2. Artifacts Per Benchmark
Every benchmark folder (`runs/benchmarks/B01..B08/`) stores: `Bxx.mp4`, `Bxx_keyframes.jpg` (360px 4-panel strip), `Bxx_plan.json`, `Bxx_engine_usage.json`, `Bxx_audio_evidence.json`, and `Bxx_review.md`. Master index: `runs/benchmarks/a2_manifest.json`.

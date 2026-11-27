# MotionLab Studio ML-01.R5 — Upstream Motion Repository Survey (Phase A2)

**Audit Date:** 2026-11-27  
**Policy:** Repository Reuse First — inspect exact upstream commits and root `LICENSE` files before copying a single line of code. Zero unverified licenses copied; zero demo footage/brand UI/remote fonts/audio bundled.

## 1. Search Scope & Catalogs Traversed
- **GitHub P0/P1 Repositories Independently Audited (20 repos):** verified live `HEAD` commit SHA, accessibility, root `LICENSE` file existence, and SHA-256 hash via GitHub API (`scripts/survey_motion_repos.py`, `scripts/inspect_upstream_licenses.py`).
- **Local Pinned HyperFrames Catalog Traversed:** `hyperframes@0.9.101 catalog --json` -> **448 catalog entries** inspected locally (100% offline-reproducible against pinned local CLI).
- **Important Disclaimer:** We did NOT examine every repository on GitHub; counts below reflect exact audited snapshots and deduplicated selections.

## 2. Complete 20-Repository Audit Matrix

| Code | Tier | Engine | Repository | Exact Audited SHA | Root LICENSE Verified | Decision | Rationale |
|---|---|---|---|---|---|---|---|
| **H01** | P0 | HyperFrames | `heygen-com/hyperframes` | `4fca68b23561` | **YES** (`Apache-2.0`, `4259155fb06f`) | **IMPORTED (9 blocks)** | Primary official HyperFrames catalog; installed 9 distinct Apache-2.0 blocks via pinned `hyperframes@0.9.101 add`, stripped CDN/audio, adapted to portrait 9:16 + local `./gsap.min.js`. |
| **H02** | P0 | Cross | `imjszhang/awesome-code-video` | `576d3b000b70` | YES (`CC0-1.0`, `d06492a8af3e`) | `DEDUPED_INDEX` | Discovery index whose HyperFrames templates fork H01; imported directly from official upstream H01 to avoid duplicate forks. |
| **H03** | P0 | HyperFrames | `chenyuxiaojin/cyxj-hyperframes` | `6a62881bb5ca` | YES (`MIT`, `a927745ec95b`) | `REFERENCE_ONLY` | Real HTML/GSAP pacing study; Chinese channel branding & tutorial assets excluded. |
| **H04** | P0 | HyperFrames | `alvin-ojoy/hyperframes` | `a89e704ffbad` | YES (`MIT`, `263623d3ac61`) | `REFERENCE_ONLY` | Brand-coupled student teaching demos; studied pacing without copying branded shots. |
| **H05** | P0 | HyperFrames | `coleam00/hyperframes-ai-video-generation` | `098505b5d6b1` | **NO** (no root LICENSE) | `BLOCKED_LICENSE` | Missing root LICENSE file; zero code copied. |
| **R01** | P0 | Remotion | `Remocn/remocn` | `52ec1ffe46df` | YES (`MIT`, `9045ed59114e`) | `DESIGN_RULES_USED` | Verified MIT; adopted craft constraints (forbidding generic halos, gradient text, same-frame card pop-ins, fake dashboards). Core mechanics covered more cleanly by R02/R06/R08/R10 without bloat. |
| **R02** | P0 | Remotion | `riaz37/remotion-ui` | `27e51ae809ea` | **YES** (`MIT`, `97d885d1d43e`) | **IMPORTED (6 primitives + 6 libs)** | Highest-value typed finance/editorial primitives: `Counter`, `WaterfallChart`, `StrikethroughReplace`, `MaskedSlideReveal`, `TrackMatte`, `ParallaxLayers` + timing/springs/chart helpers. |
| **R03** | P0 | Remotion | `GovinduSasanka/opus-motion-library` | `7bde15aa9046` | YES (`MIT`, `16a676a812b9`) | `DEDUPED_OVERLAP` | Verified MIT; chart/counter/title mechanics overlap R02 (`remotion-ui`) which has cleaner zero-dep React primitives. |
| **R04** | P0 | Remotion | `lifeprompt-team/remotion-scenes` | `02c7a84241da` | YES (`MIT`, `93a22c496702`) | `DEDUPED_OVERLAP` | Verified MIT; kinetic text/particles covered more cleanly by R02 + R06. |
| **R05** | P0 | Remotion | `av/remotion-bits` | `de35fda84b7b` | **NO** (no root LICENSE) | `BLOCKED_LICENSE` | Missing root LICENSE file at inspected SHA; zero code copied. |
| **R06** | P0 | Remotion | `vivmagarwal/remotion-effects-library` | `f2781731e10e` | **YES** (`MIT`, `941b57e5a610`) | **IMPORTED (2 primitives)** | Unique layout-rect `AspectMorphCard` (zero-squash live video reframe) + closed-form deterministic `ParticleField`. Demo footage/audio/Google Fonts stripped. |
| **R07** | P0 | Remotion | `reactvideoeditor/remotion-templates` | `6209b724798e` | **NO** (no root LICENSE) | `BLOCKED_LICENSE` | Missing root LICENSE file at inspected SHA; zero code copied. |
| **R08** | P0 | Remotion | `RenderComp/free-remotion-templates` | `f648980e528d` | **YES** (`MIT`, `3e0f0f2895a6`) | **IMPORTED (1 primitive)** | True 3D CSS `rotateY` book `FlipPageTransition` with dynamic spine lighting & cast shadow — ideal for conceptual workbook reveal. Paid catalog untouched. |
| **R09** | P0 | Remotion | `degueba/onda` | `3c8140512695` | YES (`MIT`, `14e3fb4f73fc`) | `DEDUPED_OVERLAP` | Verified MIT; typography/callout overlap with R02 + H01; avoided importing custom web fonts. |
| **R10** | P0 | Remotion | `GTKottman/remotion-transitions` | `58df89999661` | **YES** (`MIT`, `c66a3f29317e`) | **IMPORTED (2 primitives)** | Continuous `FrameMorphTransition` & `LineLedTransition`, adapted into timeline-preserving bridge wrappers (zero duration shortening). |
| **R11** | P1 | Remotion | `ali-abassi/remotion-templates` | `6430d423c417` | **NO** (no root LICENSE) | `BLOCKED_LICENSE` | Missing root LICENSE; zero code copied. |
| **R12** | P1 | Remotion | `Thedurancode/locomotion-templates` | `d42711f950c2` | **NO** (no root LICENSE) | `BLOCKED_LICENSE` | Missing root LICENSE; zero code copied. |
| **R15** | P1 | Remotion | `remotion-dev/skills` | `a3f4098a2afd` | **NO** (no root LICENSE) | `GUIDANCE_ONLY` | Official skills guidance; never counted as effects or copied. |
| **R16** | P1 | Remotion | `remotion-dev/remotion` | `19e4da7c86fa` | YES (`Remotion License`, `53e92b6560f2`) | `PINNED_RUNTIME` | Pinned local `remotion@4.0.547` runtime behind explicit `--enable-remotion` opt-in; source-available license documented in `THIRD_PARTY_NOTICES.md`. |
| **S02** | P1 | Shader | `gl-transitions/gl-transitions` | `d2efc2544ad4` | YES (`MIT`, `0db684e01505`) | `NOT_NEEDED` | Verified MIT; R10 CSS/clipPath bridges + H01 GSAP light leaks deliver deterministic CPU/GPU transitions without WebGL ANGLE portability risks. |

## 3. Summary of Imported Mechanisms (20 Distinct Across 5 Repositories)
- **Remotion (11 MIT mechanisms across 4 repos):** `rem_ru_counter`, `rem_ru_waterfall`, `rem_ru_strikethrough`, `rem_ru_masked_slide`, `rem_ru_track_matte`, `rem_ru_parallax`, `rem_viv_aspect_morph`, `rem_viv_particle_field`, `rem_rc_flip_page`, `rem_gtk_frame_morph`, `rem_gtk_line_led`.
- **HyperFrames (9 Apache-2.0 mechanisms from 1 official repo):** `hf_apple_money_count`, `hf_mk_progress_stat`, `hf_mk_specs_list`, `hf_mk_callout_highlight`, `hf_lt_mask_reveal`, `hf_strikethrough_replace`, `hf_organic_light_leak`, `hf_data_chart`, `hf_exploded_stack`.
- Full SHA-256 before/after locks and per-file patch diffs are pinned in `third_party/motion_sources.lock.json` and `motion_source_catalog.json`. Every imported directory contains verbatim upstream `LICENSE` and `NOTICE` files.


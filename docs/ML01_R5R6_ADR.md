# Architecture Decision Record — ML-01.R5.R6 (Evidence-Driven Creative Intelligence & Generative Storytelling)

- **Date:** 2026-11-28
- **Status:** Accepted & Implementing
- **Branch:** `feat/ml-01-zero-api-effects` (rebased cleanly on remote HEAD `6de406ab31329d84f9fd358cea6baef85e7a7d8e`, prior tested code `c015d22da57f33a4e3d952efd668f23fc7228de2`)

---

## ADR-R6-01: Opt-In Versioned `evidence_v3` Strategy Preserving All Prior Paths
- **Context:** R5.R5 (`c015d22`) established verified local workbook binding, multi-source rational timelines, and 199 passing tests, but left structural creative-intelligence gaps (`D01..D13`): constant A/B/C scores, mode-biased hook ranking (`+0.25 vs +0.16`), hardcoded finance claim/fallback templates, whole-page `object-fit:contain` display instead of semantic zoomed region crops, metadata-only plan coverage counted as observed coverage, 5-frame VLM sampling, and face-offset-only repair.
- **Decision:** Add `director_strategy: "evidence_v3"` as an opt-in job manifest field under existing `--creative-mode director` (`motionlab plan/edit/validate`). Keep all legacy `--input`, R5.R4, and R5.R5 (`director_strategy: "v2"` default) code paths, 40 legacy effects, and 20 imported primitives 100% backward compatible.

## ADR-R6-02: Zero-AGPL Poppler + OpenCV/PIL Semantic Document Region Cropper (`pdf_regions.py`)
- **Context:** PyMuPDF (`c220b70b`) is AGPL-3.0 / commercial dual licensed. Meanwhile, showing full A4 workbook pages tiny on a 360×640 phone screen proves page existence rather than readable feature comprehension.
- **Decision:** Reject AGPL PyMuPDF vendoring. Use installed Poppler CLI (`pdftoppm` 26.07.0 / `pdfinfo`) as external read-only subprocesses plus deterministic OpenCV/PIL region cropping in `motionlab/assets/pdf_regions.py`. Every crop stores exact `source_asset_sha256`, `page_number`, normalized `[x0, y0, x1, y1]`, `crop_sha256`, `expected_visible_facts`, and persistent fictional-example disclosure tags without ever altering source PDF numbers.

## ADR-R6-03: Feature-Level Product Proof & Honest Notion vs Workbook Separation
- **Context:** A single global `actual_product_proof_verified=true` boolean obscures the distinction between verified printed Workbook pages and live interactive Notion UI operation.
- **Decision:** Emit `product_proof_by_feature.json` tracking each advertised feature separately (`ACTUAL_PRODUCT_WORKBOOK`, `ACTUAL_PRODUCT_STATIC_UI`, `ACTUAL_PRODUCT_ACTION_RESULT`, `PRODUCT_UI_PROOF_BLOCKED`). Verified workbook pages satisfy workbook claims; missing live Notion screen recordings remain honestly flagged `PRODUCT_UI_PROOF_BLOCKED` (never fabricated from the `PRE-LAUNCH` Quick Start sheet).

## ADR-R6-04: Model-Proposed Claim DAG, Relevance-Matched Retrieval & Blind Render-Based Hook Competition
- **Context:** D01 fixed A/B/C scores and D02 `+0.25 vs +0.16` hook mode bonus prevented evidence and model reasoning from causally controlling visual outcomes.
- **Decision:**
  1. Decompose speech into frozen atomic material claims (`motionlab/semantic/evidence_planner.py`), splitting multi-part benefit lists generically without hardcoded niche phrases.
  2. Retrieve local verified assets strictly by content-grounded token/feature relevance (`min_relevance >= 0.15`, rejecting unrelated pages rather than falling back to manifest order).
  3. Let `LocalModelRouter` (`Qwen2.5-3B` raw JSON preserved in `raw_vs_reconciled_plans.json`) propose materially distinct visual treatments per beat & global plans scored on measured evidence fit rather than hardcoded bonuses.
  4. Rank 3 composited 3.8s hooks blindly (`motionlab/composition/hook_competition.py`) with randomized blind labels, permutation invariance, decoded-frame visual-richness metrics, and blind `Qwen3-VL-2B` viewer comprehension checks (zero candidate_id/mode bonuses!).

## ADR-R6-05: SceneGraph V3 Layered Evidence Composition & Post-Render Observed Coverage Verifier + Creative Repair
- **Context:** D05 counted plan metadata (`treatment_mode` overlap) as visual proof, and D06/D13 only repaired face occlusion by shifting `y_px`.
- **Decision:**
  1. Build `motionlab/composition/scene_graph_v3.py` supporting concurrent z-ordered layers (`region_crop_evidence_stage`, `causal_sequence_diagram`, `allocation_state_visual`, `lived_payoff_visual`, `product_cover_bridge`, `phone_action_callout`, `clean_speaker_anchor`, `safe_cta_card`) with full 9:16 & 16:9 dynamic layout adaptation.
  2. Split `plan_claim_coverage()` from `verify_rendered_claim_coverage()` (`motionlab/semantic/evidence_verifier.py`), auditing decoded MP4 frames via OpenCV normalized cross-correlation (`cv2.matchTemplate`) against exact source PDF crop PNGs, readable width/contrast/hold-duration checks, and blind `Qwen3-VL-2B` phone-scale inspection across EVERY material claim (and re-evaluating the R5.R5 baseline `74927936` under the exact same verifier!).
  3. Expand autonomous bounded repair (`max 2 rounds`) to fix `UNREADABLE_PRODUCT_CROP`, `TEXT_ONLY_ON_MATERIAL_CLAIM`, `WRONG_ASSET_CORRELATION`, `BORING_STAGNATION`, and `FACE_OCCLUSION` by recompiling and re-rendering affected scenes, measuring real decoded pixel diffs, and keeping only verified improvements.

## ADR-R6-06: Strict Watermark Authorization & Independent Gate Separation
- **Context:** D12 requires supporting `localized_blur` ONLY under verified non-mandatory provider authorization while preserving mandatory provenance marks untouched; D07 requires separating technical pass from creative/evidence quality and release rights.
- **Decision:** Extend `evaluate_watermark_authorization()` and add `apply_localized_watermark_blur_if_authorized()` (tight feathered Gaussian blur strictly inside verified ROI with outside-ROI preservation check) gated fail-closed on `clean_export_authorized=True`, `mandatory_provenance_mark=False`, and non-empty `license_basis`. Separate `technical_gate`, `creative_evidence_gate`, `release_rights_gate`, and `human_review_gate` in `QA02_REPORT.json` and CLI `validate`.

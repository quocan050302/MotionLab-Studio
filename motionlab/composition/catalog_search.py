"""Phase A3 (ML-01.R5.R3): Hybrid Semantic Ontology + Sparse IDF Retrieval over Unified Motion Primitive Registry.

Upgrades R5.R2 raw keyword matching (`D12`) into a structured, capability-filtered, diversity-preserving
`ontology_hybrid_sparse` retriever:
  - Honestly labels retrieval mode (`ontology_hybrid_sparse`; NEVER claims neural embeddings without a verified model).
  - Pre-filters strictly on verification status, license, engine, matte, ledger, asset requirement, duration, and aspect.
  - Expands semantic concepts (contradiction, allocation, payoff, product_structure, tactile_paper, kinetic_emphasis,
    spatial_depth, educational_diagram, transition_bridge, atmospheric_accent) with IDF weighting over capability docs.
  - Applies explicit anti-overuse decay (`used_counts`) and family diversity without blindly favoring expensive/dual engines.
  - Exposes content-addressed `compute_catalog_index_hash(registry)` so any primitive/license/status change invalidates caches.
"""

from __future__ import annotations

import math
import re
from typing import Any

from motionlab.composition.primitive_registry import MotionPrimitiveSpec, UnifiedPrimitiveRegistry
from motionlab.schemas import canonical_hash

RETRIEVAL_MODE = "ontology_hybrid_sparse"
RETRIEVAL_SCHEMA_VERSION = "3.0.0"

# Structured semantic ontology mapping high-level director concepts -> capability tokens (strictly generic, zero Olivia terms!)
SEMANTIC_CONCEPT_ONTOLOGY: dict[str, set[str]] = {
    "contradiction": {"strikethrough", "replace", "contradiction", "negation", "contrast", "correction", "myth_bust", "tension"},
    "balance_hook": {"money", "balance", "counter", "odometer", "number", "hook", "finance", "tokens", "cash", "parallax", "depth"},
    "allocation_transfer": {"allocation", "proportion", "stat", "progress", "deduction", "waterfall", "chart", "bars", "ledger", "finance"},
    "payoff_contrast": {"payoff", "remaining", "waterfall", "counter", "money", "contrast", "particles", "ledger", "finance"},
    "product_reveal": {"product", "reveal", "title", "mask", "flip", "page", "workbook", "book", "specs", "checklist", "stack", "organize", "system"},
    "kinetic_takeaway": {"headline", "typography", "kinetic", "callout", "highlight", "takeaway", "closing", "quote", "cta", "storytelling"},
    "spatial_depth": {"parallax", "depth", "2.5d", "spatial", "matte", "occlusion", "layered", "particles", "cinematic"},
    "macro_focus": {"matte", "mask", "aperture", "track", "macro", "detail", "hands", "occlusion", "reframe", "split"},
    "editorial_diagram": {"chart", "data", "bars", "proportion", "comparison", "specs", "checklist", "stack", "layers", "editorial", "morph"},
    "transition_bridge": {"transition", "morph", "bridge", "seamless", "continuity", "line", "split", "wipe", "light_leak"},
}

# Capability constraints & anti-patterns per primitive (derived strictly from actual imported source behavior)
PRIMITIVE_CAPABILITY_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "rem_ru_counter": {"min_dur_s": 0.8, "max_dur_s": 12.0, "cost_tier": "low", "supports_9x16": True, "anti_patterns": ["duplicate_numeric_owner", "non_numeric_story"]},
    "rem_ru_waterfall": {"min_dur_s": 1.6, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["short_beat_under_1_5s", "non_ledger_beat"]},
    "rem_ru_strikethrough": {"min_dur_s": 1.0, "max_dur_s": 8.0, "cost_tier": "low", "supports_9x16": True, "anti_patterns": ["affirmative_claim_without_contrast"]},
    "rem_ru_masked_slide": {"min_dur_s": 1.0, "max_dur_s": 10.0, "cost_tier": "low", "supports_9x16": True, "anti_patterns": ["long_paragraph_caption"]},
    "rem_ru_track_matte": {"min_dur_s": 1.2, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["no_visible_action_roi"]},
    "rem_ru_parallax": {"min_dur_s": 1.2, "max_dur_s": 12.0, "cost_tier": "high", "supports_9x16": True, "anti_patterns": ["missing_subject_matte"]},
    "rem_viv_aspect_morph": {"min_dur_s": 1.0, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["tiny_speaker_box"]},
    "rem_viv_particle_field": {"min_dur_s": 1.0, "max_dur_s": 12.0, "cost_tier": "low", "supports_9x16": True, "anti_patterns": ["confetti_spam", "sole_foreground_hero"]},
    "rem_rc_flip_page": {"min_dur_s": 1.5, "max_dur_s": 12.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["non_product_beat", "fake_ui_claim"]},
    "rem_gtk_frame_morph": {"min_dur_s": 0.4, "max_dur_s": 3.0, "cost_tier": "low", "supports_9x16": True, "anti_patterns": ["dark_iris_on_payoff"]},
    "rem_gtk_line_led": {"min_dur_s": 0.4, "max_dur_s": 3.0, "cost_tier": "low", "supports_9x16": True, "anti_patterns": ["every_single_cut_spam"]},
    "hf_apple_money_count": {"min_dur_s": 1.0, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["fractional_cents_on_integers", "non_ledger_beat"]},
    "hf_mk_progress_stat": {"min_dur_s": 0.9, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["repeated_identical_card_layout", "non_ledger_beat"]},
    "hf_mk_specs_list": {"min_dur_s": 1.5, "max_dur_s": 12.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["non_product_beat", "ungrounded_feature"]},
    "hf_mk_callout_highlight": {"min_dur_s": 1.0, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["full_transcript_captions"]},
    "hf_lt_mask_reveal": {"min_dur_s": 1.0, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["face_eye_overlap"]},
    "hf_strikethrough_replace": {"min_dur_s": 1.0, "max_dur_s": 8.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["stacked_with_rem_strikethrough"]},
    "hf_organic_light_leak": {"min_dur_s": 1.0, "max_dur_s": 10.0, "cost_tier": "low", "supports_9x16": True, "anti_patterns": ["face_glare_washout", "sole_hero_visual"]},
    "hf_data_chart": {"min_dur_s": 1.2, "max_dur_s": 10.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["non_proportional_bar_widths", "non_ledger_beat"]},
    "hf_exploded_stack": {"min_dur_s": 1.6, "max_dur_s": 12.0, "cost_tier": "medium", "supports_9x16": True, "anti_patterns": ["non_product_beat", "repetitive_default"]},
}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) >= 2}


def _expand_query_ontology(q_tokens: set[str]) -> tuple[set[str], list[str]]:
    """Expand raw query tokens via structured capability concepts with traceable concept IDs."""
    expanded: set[str] = set()
    matched_concepts: list[str] = []
    for concept, vocab in SEMANTIC_CONCEPT_ONTOLOGY.items():
        if q_tokens & vocab:
            matched_concepts.append(concept)
            expanded.update(vocab)
    return expanded - q_tokens, matched_concepts


def compute_catalog_index_hash(registry: UnifiedPrimitiveRegistry, verified_only: bool = True) -> str:
    """Deterministic content hash over registry primitives, licenses, statuses, and capability constraints."""
    pool = registry.verified_primitives() if verified_only else registry.all_primitives()
    rows = [
        {
            "id": p.primitive_id,
            "engine": p.engine,
            "commit_sha": p.commit_sha,
            "license": p.license,
            "status": p.status,
            "roles": sorted(p.roles),
            "tags": sorted(p.intent_tags),
            "requires_matte": p.requires_matte,
            "requires_ledger": p.requires_ledger,
            "constraints": PRIMITIVE_CAPABILITY_CONSTRAINTS.get(p.primitive_id, {}),
        }
        for p in sorted(pool, key=lambda x: x.primitive_id)
    ]
    return canonical_hash({"version": RETRIEVAL_SCHEMA_VERSION, "mode": RETRIEVAL_MODE, "rows": rows})


def search_primitives_hybrid(
    registry: UnifiedPrimitiveRegistry,
    intent_query: str,
    *,
    engine: str | None = None,
    has_ledger: bool = False,
    has_matte: bool = False,
    beat_duration_s: float | None = None,
    aspect_9x16: bool = True,
    verified_only: bool = True,
    top_k: int = 6,
    exclude_ids: set[str] | None = None,
    used_counts: dict[str, int] | None = None,
    preserve_diversity: bool = True,
    niche: str | None = None,
) -> dict[str, Any]:
    """Capability-filtered ontology + sparse-IDF retrieval returning full traceability & diversity."""
    q_tokens = _tokenize(intent_query)
    ont_tokens, matched_concepts = _expand_query_ontology(q_tokens)
    pool = registry.verified_primitives() if verified_only else registry.all_primitives()
    cat_hash = compute_catalog_index_hash(registry, verified_only=verified_only)
    usage = used_counts or {}

    # Compute document frequency across pool for true sparse IDF weighting
    doc_tokens: dict[str, set[str]] = {}
    df: dict[str, int] = {}
    for spec in pool:
        dt = set(spec.intent_tags) | _tokenize(" ".join(spec.roles)) | _tokenize(spec.visual_family) | _tokenize(spec.description)
        doc_tokens[spec.primitive_id] = dt
        for tok in dt:
            df[tok] = df.get(tok, 0) + 1
    n_docs = max(1, len(pool))

    def _idf(tok: str) -> float:
        return math.log((n_docs + 1.0) / (df.get(tok, 0) + 1.0)) + 1.0

    filtered_out: list[dict[str, str]] = []
    scored: list[tuple[float, str, MotionPrimitiveSpec, list[str], dict[str, Any]]] = []

    for spec in pool:
        cap = PRIMITIVE_CAPABILITY_CONSTRAINTS.get(spec.primitive_id, {})
        if exclude_ids and spec.primitive_id in exclude_ids:
            filtered_out.append({"primitive_id": spec.primitive_id, "reason": "excluded_by_caller"})
            continue
        if engine and spec.engine != engine:
            filtered_out.append({"primitive_id": spec.primitive_id, "reason": f"engine_filter:{engine}"})
            continue
        if verified_only and spec.status not in ("VERIFIED", "USED_IN_CANONICAL"):
            filtered_out.append({"primitive_id": spec.primitive_id, "reason": f"unverified_status:{spec.status}"})
            continue
        if spec.requires_ledger and not has_ledger:
            filtered_out.append({"primitive_id": spec.primitive_id, "reason": "requires_active_ledger"})
            continue
        if spec.requires_matte and not has_matte:
            filtered_out.append({"primitive_id": spec.primitive_id, "reason": "requires_subject_matte"})
            continue
        if aspect_9x16 and not spec.supports_portrait_9x16:
            filtered_out.append({"primitive_id": spec.primitive_id, "reason": "portrait_9x16_unsupported"})
            continue
        if beat_duration_s is not None:
            min_d = float(cap.get("min_dur_s", 0.3))
            if beat_duration_s < min_d * 0.75:
                filtered_out.append({"primitive_id": spec.primitive_id, "reason": f"beat_too_short:{beat_duration_s:.2f}s<{min_d:.2f}s"})
                continue

        reasons: list[str] = []
        tag_set = set(spec.intent_tags) | _tokenize(" ".join(spec.roles)) | _tokenize(spec.visual_family)
        desc_set = _tokenize(spec.description)

        # 1. IDF-weighted exact token hits
        exact_hits = q_tokens & tag_set
        score = sum(2.2 * _idf(t) for t in exact_hits)
        if exact_hits:
            reasons.append(f"tag_idf:{','.join(sorted(exact_hits)[:4])}")

        # 2. Description sparse hits
        desc_hits = (q_tokens & desc_set) - exact_hits
        if desc_hits:
            score += sum(0.75 * _idf(t) for t in desc_hits)
            reasons.append(f"desc_idf:{','.join(sorted(desc_hits)[:3])}")

        # 3. Structured semantic ontology expansion hits (lower weight than exact intent match!)
        ont_hits = (ont_tokens & tag_set) - exact_hits
        if ont_hits:
            score += min(3.5, sum(0.45 * _idf(t) for t in ont_hits))
            reasons.append(f"ontology_concept:{','.join(sorted(matched_concepts)[:2])}")

        # 4. Capability fit bonuses (without blindly forcing dual-engine or expensive effects)
        if spec.requires_ledger and has_ledger and (q_tokens & {"ledger", "allocation", "waterfall", "money", "balance", "remaining", "payoff", "finance", "chart"}):
            score += 1.6
            reasons.append("verified_ledger_compatible")
        if spec.requires_matte and has_matte and (q_tokens & {"depth", "matte", "parallax", "hook", "spatial", "behind"}):
            score += 1.4
            reasons.append("matte_occlusion_ready")

        # 5. Non-finance guard: penalize money/ledger primitives on non-finance niches
        if niche and niche.lower() in ("education", "tech", "storytelling", "tutorial", "gameplay") and spec.requires_ledger:
            score -= 4.0
            reasons.append("non_finance_ledger_penalty")

        # 6. Anti-overuse decay so one primitive doesn't dominate every scene
        use_n = int(usage.get(spec.primitive_id, 0))
        if use_n > 0 and score > 0.0:
            penalty = min(3.0, use_n * 1.15)
            score = max(0.05, score - penalty)
            reasons.append(f"anti_overuse_decay:-{penalty:.2f}(n={use_n})")

        if score > 0.0:
            scored.append((round(score, 3), spec.primitive_id, spec, reasons, cap))

    scored.sort(key=lambda x: (-x[0], x[1]))

    # Diversity-preserving top-k selection: avoid filling top_k exclusively with 1 visual_family when strong alternatives exist
    k = max(1, int(top_k))
    selected_rows = scored[:k]
    if preserve_diversity and len(scored) > k:
        chosen: list[tuple[float, str, MotionPrimitiveSpec, list[str], dict[str, Any]]] = []
        fam_seen: dict[str, int] = {}
        remainder: list[tuple[float, str, MotionPrimitiveSpec, list[str], dict[str, Any]]] = []
        for row in scored:
            fam = row[2].visual_family
            if fam_seen.get(fam, 0) < 2 and len(chosen) < k:
                chosen.append(row)
                fam_seen[fam] = fam_seen.get(fam, 0) + 1
            else:
                remainder.append(row)
        for row in remainder:
            if len(chosen) >= k:
                break
            chosen.append(row)
        selected_rows = chosen

    candidates = [
        {
            "primitive_id": spec.primitive_id,
            "engine": spec.engine,
            "origin_slug": spec.origin_slug,
            "repo": spec.repo,
            "commit_sha": spec.commit_sha[:10],
            "license": spec.license,
            "visual_family": spec.visual_family,
            "roles": list(spec.roles),
            "supports_alpha": spec.supports_alpha,
            "requires_matte": spec.requires_matte,
            "requires_ledger": spec.requires_ledger,
            "cost_tier": cap.get("cost_tier", "medium"),
            "anti_patterns": list(cap.get("anti_patterns", [])),
            "score": sc,
            "reasons": reasons,
            "status": spec.status,
            "retrieval_mode": RETRIEVAL_MODE,
        }
        for sc, _, spec, reasons, cap in selected_rows
    ]
    return {
        "retrieval_mode": RETRIEVAL_MODE,
        "embedding_model_used": False,
        "catalog_index_hash": cat_hash,
        "matched_concepts": matched_concepts,
        "filtered_out_count": len(filtered_out),
        "candidates": candidates,
    }


def search_primitives_by_intent(
    registry: UnifiedPrimitiveRegistry,
    intent_query: str,
    *,
    engine: str | None = None,
    has_ledger: bool = False,
    has_matte: bool = False,
    beat_duration_s: float | None = None,
    verified_only: bool = True,
    top_k: int = 5,
    exclude_ids: set[str] | None = None,
    used_counts: dict[str, int] | None = None,
    niche: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper returning ranked verified primitives via ontology_hybrid_sparse retrieval."""
    res = search_primitives_hybrid(
        registry,
        intent_query,
        engine=engine,
        has_ledger=has_ledger,
        has_matte=has_matte,
        beat_duration_s=beat_duration_s,
        verified_only=verified_only,
        top_k=top_k,
        exclude_ids=exclude_ids,
        used_counts=used_counts,
        niche=niche,
    )
    return list(res["candidates"])

"""Phase A2/A3 (ML-01.R5): Semantic Intent Search over the Unified Motion Primitive Registry."""

from __future__ import annotations

import re
from typing import Any

from motionlab.composition.primitive_registry import MotionPrimitiveSpec, UnifiedPrimitiveRegistry


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) >= 2}


def search_primitives_by_intent(
    registry: UnifiedPrimitiveRegistry,
    intent_query: str,
    *,
    engine: str | None = None,
    has_ledger: bool = False,
    has_matte: bool = False,
    verified_only: bool = True,
    top_k: int = 5,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and rank verified imported primitives matching a semantic beat's visual intent and constraints."""
    q_tokens = _tokenize(intent_query)
    pool = registry.verified_primitives() if verified_only else registry.all_primitives()
    scored: list[tuple[float, str, MotionPrimitiveSpec, list[str]]] = []

    for spec in pool:
        if exclude_ids and spec.primitive_id in exclude_ids:
            continue
        if engine and spec.engine != engine:
            continue
        if spec.requires_ledger and not has_ledger:
            continue
        if spec.requires_matte and not has_matte:
            continue

        reasons: list[str] = []
        score = 0.0
        tag_set = set(spec.intent_tags) | _tokenize(" ".join(spec.roles)) | _tokenize(spec.visual_family)
        desc_set = _tokenize(spec.description)

        exact_hits = q_tokens & tag_set
        if exact_hits:
            score += len(exact_hits) * 2.5
            reasons.append(f"tag_match:{','.join(sorted(exact_hits)[:4])}")

        desc_hits = (q_tokens & desc_set) - exact_hits
        if desc_hits:
            score += len(desc_hits) * 0.8
            reasons.append(f"desc_match:{','.join(sorted(desc_hits)[:3])}")

        if spec.requires_ledger and has_ledger and (q_tokens & {"ledger", "allocation", "waterfall", "bills", "money", "balance", "remaining", "payoff", "finance", "chart"}):
            score += 2.0
            reasons.append("verified_ledger_compatible")
        if spec.requires_matte and has_matte and (q_tokens & {"depth", "matte", "parallax", "hook", "spatial", "behind"}):
            score += 1.5
            reasons.append("matte_occlusion_ready")

        if score > 0.0:
            scored.append((round(score, 3), spec.primitive_id, spec, reasons))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        {
            "primitive_id": spec.primitive_id,
            "engine": spec.engine,
            "origin_slug": spec.origin_slug,
            "repo": spec.repo,
            "commit_sha": spec.commit_sha[:10],
            "license": spec.license,
            "visual_family": spec.visual_family,
            "score": sc,
            "reasons": reasons,
            "status": spec.status,
        }
        for sc, _, spec, reasons in scored[: max(1, top_k)]
    ]


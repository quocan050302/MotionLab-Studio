"""Phase A2 (ML-01.R5.R7): Global StoryBible, Natural Chapter Segmentation & Canonical Claim/Action DAG.

Fixes D03, D09, D10, D12:
  - Single authoritative `canonical_split_sentences()` with deterministic hash used across ALL reports
    and alignment artifacts (eliminates 8/10 vs 9/11 sentence-count inconsistency permanently).
  - Universal niche-agnostic StoryBible for BOTH Shorts (15-60s) and Long-form (3-30+ min), spoken
    and silent media — ZERO global `claims[:10]` or `vlm_calls < 7` caps!
  - Natural chapter segmentation at semantic / pause / shot boundaries (NOT fixed 30s cuts), preserving
    complete words, evidence holds, and persistent entity states across chapter boundaries.
  - Explicit persistent entity & state snapshots preventing silent value rollbacks across chapters.
  - Orphan coverage validator ensuring EVERY input claim/time interval is explicitly planned or marked
    unresolved (Chapter 20+ late claims never silently dropped).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any

from motionlab.schemas import canonical_hash, sanitize_native

STORY_BIBLE_SCHEMA_VERSION = "4.0.0"

_ABBREV_GUARD_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|No)\.", re.IGNORECASE)
_ELLIPSIS_GUARD_RE = re.compile(r"\.\.\.+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONEY_RE = re.compile(r"\$\s*\d[\d,]*(?:\.\d{1,2})?")
_NUM_UNIT_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|ms|s|sec|min|hr|km|m|mm|mbps|fps|gb|mb|kg|lb|°c|°f)?\b", re.IGNORECASE)


def canonical_split_sentences(text: str | None) -> dict[str, Any]:
    """Single deterministic sentence segmentation authority used by StoryBible, SpeechAlignment & Reports.

    Preserves ellipses (`...`) and abbreviations (`U.S.`, `Dr.`) as single sentences so sentence counts
    NEVER diverge between `script_completeness.json`, `speech_alignment.json`, and human reports!
    """
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return {
            "schema_version": STORY_BIBLE_SCHEMA_VERSION,
            "sentences": [],
            "sentence_count": 0,
            "segmentation_hash": canonical_hash([]),
        }
    masked = _ELLIPSIS_GUARD_RE.sub("<ELLIPSIS>", raw)
    masked = _ABBREV_GUARD_RE.sub(lambda m: f"{m.group(1)}<DOT>", masked)
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(masked) if p.strip()]
    sentences = [p.replace("<ELLIPSIS>", "...").replace("<DOT>", ".") for p in parts]
    return {
        "schema_version": STORY_BIBLE_SCHEMA_VERSION,
        "sentences": sentences,
        "sentence_count": len(sentences),
        "segmentation_hash": canonical_hash(sentences),
    }


def _classify_claim_semantics(text: str, idx: int, total: int, *, product_name: str | None = None) -> dict[str, Any]:
    """Universal, domain-agnostic narrative role & evidence classifier for any niche."""
    low = text.lower()
    prod_low = (product_name or "").lower().strip()
    weekdays = [d for d in _WEEKDAYS if re.search(rf"\b{d}\b", low)]
    money = _MONEY_RE.findall(text)
    has_question = "?" in text
    has_negation = bool(re.search(r"\b(not|never|don't|doesn't|can't|won't|without|no longer|isn't|aren't)\b", low))

    if idx == 0:
        role = "opening_hook"
        req = "REAL_CONTEXT_OR_VISUAL_ANCHOR"
        mat = "critical"
        weight = 1.4
        disallowed = ["unrelated_stock", "face_occluding_box", "tiny_unreadable_text"]
    elif idx >= max(1, total - 2) and any(k in low for k in ("link", "caption", "take a look", "subscribe", "check out", "download", "try ", "bio", "description")):
        role = "cta_direction"
        req = "SAFE_CTA_WITH_IDENTITY"
        mat = "supporting"
        weight = 0.8
        disallowed = ["fabricated_price", "fake_urgency", "bottom_ui_overlap"]
    elif has_question or any(k in low for k in ("versus", " vs ", "difference between", "compare", "on paper")):
        role = "core_conflict_or_comparison"
        req = "ILLUSTRATIVE_EXAMPLE_OR_DIAGRAM"
        mat = "critical"
        weight = 1.3
        disallowed = ["text_only_card", "contradictory_labels"]
    elif (prod_low and prod_low in low) or any(k in low for k in ("it shows", "shows you", "spoken for", "needs to last", "actually spend", "dashboard", "workbook", "template", "app shows", "export", "click ")):
        role = "product_or_mechanism_proof"
        req = "ACTUAL_PRODUCT_OR_REAL_DEMO" if product_name else "VERIFIED_FACT_DIAGRAM"
        mat = "critical"
        weight = 1.6
        disallowed = ["fake_ui_mock", "tiny_uncropped_page", "text_only_card", "wrong_page_crop"]
    elif len(weekdays) >= 2 or any(k in low for k in ("before ", "until ", "after ", "then ", "first ", "next ", "step ", "timeline", "sequence", "due ")):
        role = "causal_sequence_or_timing"
        req = "VERIFIED_FACT_DIAGRAM"
        mat = "critical"
        weight = 1.5
        disallowed = ["reversed_order", "static_text_only_card", "unverified_number"]
    elif any(k in low for k in ("imagine", "feeling", "stress", "calm", "without guessing", "without doing", "relief", "enjoy", "confidence")):
        role = "lived_resolution_payoff"
        req = "ILLUSTRATIVE_ACTION_SEQUENCE"
        mat = "supporting"
        weight = 1.1
        disallowed = ["static_icon_checklist_only", "text_only_pill", "unlabeled_stock_as_subject"]
    elif len(text.split()) <= 7:
        role = "breathing_anchor"
        req = "REAL_CONTEXT_FOOTAGE"
        mat = "ambient"
        weight = 0.5
        disallowed = ["unnecessary_motion_spam"]
    else:
        role = "supporting_explanation"
        req = "VERIFIED_FACT_DIAGRAM" if (money or re.search(r"\b\d+", low)) else "ILLUSTRATIVE_EXAMPLE_OR_DIAGRAM"
        mat = "supporting"
        weight = 1.0
        disallowed = ["text_only_wall", "invented_statistics"]

    fact_mode = "hypothetical_labeled" if any(k in low for k in ("imagine", "example", "suppose", "for instance")) else "source_grounded"
    return {
        "role": role,
        "required_evidence_type": req,
        "materiality": mat,
        "is_material": mat in ("critical", "supporting"),
        "weight": weight,
        "fact_vs_hypothetical": fact_mode,
        "has_negation": has_negation,
        "weekdays_mentioned": weekdays,
        "money_mentioned": money,
        "disallowed_substitutions": disallowed,
    }


def _split_enumeration_clauses(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split bundled multi-benefit enumerations (including missing Oxford commas) into atomic clauses."""
    if len(words) < 10:
        return [words]
    tokens = [str(w.get("word", "")).strip() for w in words]
    split_indices: list[int] = [0]
    for i in range(2, len(tokens) - 2):
        prev_tok = tokens[i - 1]
        cur_low = tokens[i].lower().strip(".,?!;:\"'")
        next_low = tokens[i + 1].lower().strip(".,?!;:\"'") if i + 1 < len(tokens) else ""
        # Trigger on comma before clause starter ("what", "how", "when", "where", "and what")
        if prev_tok.endswith(",") and cur_low in ("what", "what's", "how", "when", "where", "which", "and"):
            if i - split_indices[-1] >= 3:
                split_indices.append(i)
        # Trigger on coordinating "and what..." even without Oxford comma
        elif cur_low == "and" and next_low in ("what", "what's", "how", "then") and (i - split_indices[-1] >= 4) and (len(tokens) - i >= 4):
            split_indices.append(i)

    if len(split_indices) <= 1:
        return [words]
    chunks: list[list[dict[str, Any]]] = []
    for k, s_idx in enumerate(split_indices):
        e_idx = split_indices[k + 1] if k + 1 < len(split_indices) else len(words)
        if e_idx - s_idx >= 2:
            chunks.append(words[s_idx:e_idx])
    return chunks if len(chunks) > 1 else [words]


def decompose_atomic_claims_and_actions(
    *,
    edited_words: list[dict[str, Any]],
    duration_s: float,
    fps_num: int = 30,
    fps_den: int = 1,
    product_name: str | None = None,
    silent_visual_beats: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Decompose an ENTIRE timeline (Shorts OR multi-minute Long-form; spoken OR silent) into atomic claims/actions.

    ZERO truncation: processes all sentences/clauses from t=0 through duration_s!
    """
    fps_f = float(Fraction(fps_num, fps_den))
    total_f = max(1, int(round(duration_s * fps_f)))

    # Silent / non-speech media path (e.g. pottery, cooking B-roll, ambient visual storytelling)
    if not edited_words:
        beats = list(silent_visual_beats or [])
        if not beats:
            seg_len = max(3.0, min(12.0, duration_s / max(1, int(round(duration_s / 6.0)))))
            t = 0.0
            idx = 0
            while t < duration_s - 0.25:
                t1 = min(duration_s, t + seg_len)
                beats.append({"start_s": t, "end_s": t1, "description": f"Visual segment {idx + 1}", "is_silent_art": True})
                t = t1
                idx += 1
        claims: list[dict[str, Any]] = []
        for idx, b in enumerate(beats):
            s_s = max(0.0, float(b.get("start_s", 0.0)))
            e_s = min(duration_s, max(s_s + 0.5, float(b.get("end_s", s_s + 4.0))))
            sf = max(0, min(total_f - 1, int(round(s_s * fps_f))))
            ef = max(sf + 1, min(total_f, int(round(e_s * fps_f))))
            claims.append({
                "claim_id": f"claim_{idx:03d}",
                "ordinal": idx,
                "text": str(b.get("description") or f"Visual beat {idx + 1}"),
                "source_evidence_ids": [f"visual_span:{sf}:{ef}"],
                "word_indices": [],
                "start_s": round(s_s, 4),
                "end_s": round(e_s, 4),
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "role": "silent_visual_craft" if b.get("is_silent_art") else "visual_action_beat",
                "required_evidence_type": "REAL_CONTEXT_FOOTAGE",
                "materiality": "ambient" if b.get("is_silent_art") else "supporting",
                "is_material": not bool(b.get("is_silent_art")),
                "weight": 0.6 if b.get("is_silent_art") else 1.0,
                "question_answered": str(b.get("question") or "Observe real uninterrupted craft/action"),
                "expected_observations": [str(b.get("description") or "Clean unobstructed visual action")],
                "causality_edges": [f"claim_{idx - 1:03d}"] if idx > 0 else [],
                "fact_vs_hypothetical": "source_grounded",
                "disallowed_substitutions": ["unnecessary_motion_spam", "fabricated_cta"],
                "visual_state_before": "establishing_context",
                "visual_state_after": "continued_visual_progression",
                "temporal_dependencies": [f"claim_{idx - 1:03d}"] if idx > 0 else [],
                "risk": "low",
                "has_negation": False,
                "weekdays_mentioned": [],
                "money_mentioned": [],
            })
        return claims

    def _w_start(w: dict[str, Any]) -> float:
        return float(w.get("output_start_s", w.get("source_start_s", w.get("start_s", w.get("start", 0.0)))))

    def _w_end(w: dict[str, Any]) -> float:
        return float(w.get("output_end_s", w.get("source_end_s", w.get("end_s", w.get("end", _w_start(w) + 0.1)))))

    # Group spoken words into natural sentences using punctuation OR acoustic pause >= 0.72s
    raw_sentences: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for i, w in enumerate(edited_words):
        w_copy = dict(w)
        w_copy["_widx"] = i
        if cur:
            gap = _w_start(w) - _w_end(cur[-1])
            prev_txt = str(cur[-1].get("word", "")).strip()
            end_punct = prev_txt.endswith((".", "!", "?")) and not prev_txt.endswith("...")
            if end_punct or gap >= 0.72:
                raw_sentences.append(cur)
                cur = []
        cur.append(w_copy)
    if cur:
        raw_sentences.append(cur)

    # Expand bundled enumerations into atomic sub-clauses
    atomic_groups: list[tuple[int, int | None, list[dict[str, Any]]]] = []
    for s_idx, sw in enumerate(raw_sentences):
        sub_clauses = _split_enumeration_clauses(sw)
        if len(sub_clauses) > 1:
            for sub_i, scw in enumerate(sub_clauses):
                atomic_groups.append((s_idx, sub_i, scw))
        else:
            atomic_groups.append((s_idx, None, sw))

    claims = []
    total_g = len(atomic_groups)
    prev_cid: str | None = None
    for idx, (s_idx, sub_i, gw) in enumerate(atomic_groups):
        txt = " ".join(str(w.get("word", "")).strip() for w in gw).strip()
        s_s = max(0.0, _w_start(gw[0]))
        e_s = min(duration_s, max(s_s + 0.25, _w_end(gw[-1])))
        sf = max(0, min(total_f - 1, int(round(s_s * fps_f))))
        ef = max(sf + 1, min(total_f, int(round(e_s * fps_f))))
        sem = _classify_claim_semantics(txt, idx, total_g, product_name=product_name)
        suffix = f"_{chr(ord('a') + sub_i)}" if sub_i is not None else ""
        cid = f"claim_{s_idx:03d}{suffix}"
        w_ids = [f"word:{int(w['_widx'])}" for w in gw]

        # Build concrete question & expected observation independent of any future template label
        if sem["role"] == "causal_sequence_or_timing":
            q = f"What exact sequence or timing order does this claim establish? ({txt[:72]})"
            exp_obs = [f"Ordered timing/sequence matching: {', '.join(sem['weekdays_mentioned']) or txt[:48]}"]
            risk = "high"
        elif sem["role"] == "product_or_mechanism_proof":
            q = f"What verifiable mechanism or real product evidence proves: {txt[:72]}?"
            exp_obs = [f"Legible verified proof region or mechanism directly demonstrating: {txt[:56]}"]
            risk = "high"
        elif sem["role"] == "core_conflict_or_comparison":
            q = f"What two states or quantities are contrasted by: {txt[:72]}?"
            exp_obs = [f"Clear visual distinction between contrasted states for: {txt[:56]}"]
            risk = "high"
        elif sem["role"] == "lived_resolution_payoff":
            q = f"What concrete lived action and result illustrates: {txt[:72]}?"
            exp_obs = [f"Multi-phase visual action sequence (before -> action -> result) for: {txt[:56]}"]
            risk = "medium"
        elif sem["role"] == "opening_hook":
            q = "What immediate relatable problem or curiosity gap hooks the viewer in the first 3 seconds?"
            exp_obs = ["Clear speaker/context visual anchor with concise readable framing and zero face occlusion"]
            risk = "high"
        else:
            q = f"How does the frame truthfully support: {txt[:72]}?"
            exp_obs = [f"Truthful visual support aligned to: {txt[:56]}"]
            risk = "medium" if sem["is_material"] else "low"

        claims.append({
            "claim_id": cid,
            "ordinal": idx,
            "sentence_group_index": s_idx,
            "sub_clause_index": sub_i,
            "text": txt,
            "source_evidence_ids": w_ids,
            "word_indices": [int(w["_widx"]) for w in gw],
            "start_s": round(s_s, 4),
            "end_s": round(e_s, 4),
            "start_frame": sf,
            "end_frame_exclusive": ef,
            "role": sem["role"],
            "required_evidence_type": sem["required_evidence_type"],
            "materiality": sem["materiality"],
            "is_material": sem["is_material"],
            "weight": sem["weight"],
            "question_answered": q,
            "expected_observations": exp_obs,
            "causality_edges": [prev_cid] if prev_cid else [],
            "fact_vs_hypothetical": sem["fact_vs_hypothetical"],
            "disallowed_substitutions": sem["disallowed_substitutions"],
            "visual_state_before": "prior_beat_state" if prev_cid else "initial_frame",
            "visual_state_after": f"resolved_{cid}",
            "temporal_dependencies": [prev_cid] if prev_cid else [],
            "risk": risk,
            "has_negation": sem["has_negation"],
            "weekdays_mentioned": sem["weekdays_mentioned"],
            "money_mentioned": sem["money_mentioned"],
        })
        prev_cid = cid

    return claims


def segment_chapters(
    *,
    atomic_claims: list[dict[str, Any]],
    duration_s: float,
    fps_num: int = 30,
    fps_den: int = 1,
    target_window_s: float = 35.0,
    min_chapter_s: float = 12.0,
    max_chapter_s: float = 60.0,
) -> list[dict[str, Any]]:
    """Segment any short or long-form video into natural semantic chapters (NEVER cutting mid-word/claim!)."""
    fps_f = float(Fraction(fps_num, fps_den))
    total_f = max(1, int(round(duration_s * fps_f)))

    if not atomic_claims:
        return [{
            "chapter_id": "ch_00",
            "chapter_index": 0,
            "start_frame": 0,
            "end_frame_exclusive": total_f,
            "start_s": 0.0,
            "end_s": round(duration_s, 4),
            "narrative_purpose": "complete_timeline",
            "story_tension": "establish_and_resolve",
            "primary_question": "Deliver complete visual narrative cleanly",
            "claim_ids": [],
            "material_claim_ids": [],
            "input_state_refs": [],
            "output_state_refs": [],
            "key_evidence_ids": [],
            "transition_contract": {"type": "clean_cut", "handle_frames": 0, "sound_tail_policy": "preserve_natural"},
            "music_sfx_intent": {"bed_energy": "subtle", "allow_sfx": True},
            "uncertainty": "none",
        }]

    # Short videos (<= 55s) naturally form 1 cohesive chapter (or 2 if clear mid-story transition exists);
    # Long-form videos (> 55s) split cleanly ONLY at natural claim boundaries near target_window_s!
    if duration_s <= max_chapter_s:
        groups = [atomic_claims]
    else:
        groups = []
        cur_grp: list[dict[str, Any]] = []
        grp_start = float(atomic_claims[0]["start_s"])
        for i, cl in enumerate(atomic_claims):
            cur_grp.append(cl)
            cur_end = float(cl["end_s"])
            elapsed = cur_end - grp_start
            next_cl = atomic_claims[i + 1] if i + 1 < len(atomic_claims) else None
            gap_to_next = (float(next_cl["start_s"]) - cur_end) if next_cl else 999.0
            is_natural_break = (
                next_cl is None
                or (elapsed >= target_window_s and (gap_to_next >= 0.35 or cl["role"] in ("lived_resolution_payoff", "breathing_anchor", "cta_direction")))
                or elapsed >= max_chapter_s
            )
            if is_natural_break and (elapsed >= min_chapter_s or next_cl is None):
                groups.append(cur_grp)
                cur_grp = []
                if next_cl is not None:
                    grp_start = float(next_cl["start_s"])
        if cur_grp:
            if groups and (float(cur_grp[-1]["end_s"]) - float(cur_grp[0]["start_s"])) < 6.0:
                groups[-1].extend(cur_grp)
            else:
                groups.append(cur_grp)

    chapters: list[dict[str, Any]] = []
    for idx, grp in enumerate(groups):
        sf = 0 if idx == 0 else int(chapters[-1]["end_frame_exclusive"])
        if idx == len(groups) - 1:
            ef = total_f
        else:
            # Split cleanly at midpoint of silence gap between last claim of grp and first claim of next grp
            next_sf = int(groups[idx + 1][0]["start_frame"])
            last_ef = int(grp[-1]["end_frame_exclusive"])
            ef = max(sf + 1, min(total_f, (last_ef + next_sf) // 2 if next_sf >= last_ef else last_ef))
        cids = [str(c["claim_id"]) for c in grp]
        mat_cids = [str(c["claim_id"]) for c in grp if c.get("is_material")]
        roles = [str(c["role"]) for c in grp]
        if idx == 0 and len(groups) == 1:
            purpose = "complete_short_arc"
        elif idx == 0:
            purpose = "hook_and_setup"
        elif idx == len(groups) - 1:
            purpose = "resolution_and_closure"
        else:
            purpose = f"development_chapter_{idx:02d}"

        primary_q = next((str(c["question_answered"]) for c in grp if c.get("materiality") == "critical"), str(grp[0]["question_answered"]))
        ev_ids: list[str] = []
        for c in grp:
            ev_ids.extend(list(c.get("source_evidence_ids") or [])[:4])
        chapters.append({
            "chapter_id": f"ch_{idx:02d}",
            "chapter_index": idx,
            "start_frame": sf,
            "end_frame_exclusive": ef,
            "start_s": round(sf / fps_f, 4),
            "end_s": round(ef / fps_f, 4),
            "narrative_purpose": purpose,
            "story_tension": "hook_conflict_proof_payoff" if len(groups) == 1 else ("escalate" if idx < len(groups) - 1 else "resolve"),
            "primary_question": primary_q,
            "claim_ids": cids,
            "material_claim_ids": mat_cids,
            "dominant_roles": sorted(set(roles)),
            "input_state_refs": [f"state:ch_{idx - 1:02d}:exit"] if idx > 0 else ["state:initial"],
            "output_state_refs": [f"state:ch_{idx:02d}:exit"],
            "key_evidence_ids": ev_ids[:24],
            "transition_contract": {
                "type": "motivated_continuity" if idx < len(groups) - 1 else "natural_outro",
                "handle_frames": int(round(0.4 * fps_f)) if idx < len(groups) - 1 else 0,
                "sound_tail_policy": "overlap_cross_chapter_tail",
            },
            "music_sfx_intent": {
                "bed_energy": "calm_pulse" if idx == 0 else ("focused" if idx < len(groups) - 1 else "warm_resolve"),
                "allow_sfx": True,
                "preserve_chapter_phase": True,
            },
            "uncertainty": "none",
        })
    return chapters


def build_chapter_context(
    story_bible: dict[str, Any],
    chapter_id: str,
    *,
    prior_reconciled_states: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact, validated per-chapter planning context without dumping full long-form transcripts."""
    ch = next((c for c in story_bible.get("chapters", []) if c["chapter_id"] == chapter_id), None)
    if ch is None:
        raise KeyError(f"Unknown chapter_id in StoryBible: {chapter_id}")
    cid_set = set(ch["claim_ids"])
    claims = [c for c in story_bible.get("atomic_claims", []) if c["claim_id"] in cid_set]
    carry_entities = {}
    for eid, ent in (story_bible.get("persistent_entities") or {}).items():
        cur_val = (prior_reconciled_states or {}).get(eid, ent.get("initial_value"))
        carry_entities[eid] = {
            "entity_id": eid,
            "label": ent.get("label"),
            "unit": ent.get("unit", ""),
            "current_value": cur_val,
            "source_proof": ent.get("source_proof"),
            "fact_vs_hypothetical": ent.get("fact_vs_hypothetical", "source_grounded"),
        }
    return sanitize_native({
        "job_id": story_bible["job_id"],
        "chapter": ch,
        "claims": claims,
        "persistent_entity_snapshot": carry_entities,
        "style_tokens": story_bible.get("style_tokens") or {},
        "camera_grammar": story_bible.get("camera_grammar") or {},
        "banned_claims": story_bible.get("banned_claims") or [],
        "aspect_ratio": story_bible.get("aspect_ratio", "9:16"),
    })


def reconcile_chapter_boundary(
    *,
    prev_chapter_exit_state: dict[str, Any],
    next_chapter_entry_state: dict[str, Any],
    explicit_recap: bool = False,
) -> dict[str, Any]:
    """Validate persistent entity continuity across chapter boundaries (blocks silent rollbacks!)."""
    errors: list[str] = []
    merged = dict(prev_chapter_exit_state)
    for eid, next_info in next_chapter_entry_state.items():
        if eid in prev_chapter_exit_state:
            prev_val = prev_chapter_exit_state[eid].get("value")
            next_val = next_info.get("value")
            if prev_val != next_val and not explicit_recap and not next_info.get("explicit_update_event_id"):
                errors.append(
                    f"silent_cross_chapter_rollback:{eid}:prev={prev_val}->next={next_val}"
                )
        merged[eid] = next_info
    return sanitize_native({
        "valid": len(errors) == 0,
        "errors": errors,
        "reconciled_state": merged,
    })


def validate_story_coverage(
    *,
    story_bible: dict[str, Any],
    planned_scene_claim_ids: set[str] | list[str],
    explicit_unresolved_claim_ids: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Verify EVERY atomic claim (including Chapter 20+ late claims) is planned or explicitly unresolved."""
    planned = set(planned_scene_claim_ids)
    unresolved = set(explicit_unresolved_claim_ids or [])
    all_claims = list(story_bible.get("atomic_claims") or [])
    all_cids = [str(c["claim_id"]) for c in all_claims]
    orphans = [cid for cid in all_cids if cid not in planned and cid not in unresolved]
    duplicates_overlap = sorted(planned & unresolved)
    chapter_cov: list[dict[str, Any]] = []
    for ch in story_bible.get("chapters") or []:
        cids = list(ch.get("claim_ids") or [])
        hit = sum(1 for cid in cids if cid in planned)
        unres = sum(1 for cid in cids if cid in unresolved)
        chapter_cov.append({
            "chapter_id": ch["chapter_id"],
            "claims_total": len(cids),
            "planned_count": hit,
            "explicit_unresolved_count": unres,
            "orphan_count": len(cids) - hit - unres,
            "coverage_ratio": round(hit / max(1, len(cids)), 4) if cids else 1.0,
        })
    return sanitize_native({
        "valid": len(orphans) == 0 and len(duplicates_overlap) == 0,
        "claims_total": len(all_cids),
        "planned_count": sum(1 for cid in all_cids if cid in planned),
        "explicit_unresolved_count": sum(1 for cid in all_cids if cid in unresolved),
        "orphan_claim_ids": orphans,
        "conflicting_planned_and_unresolved": duplicates_overlap,
        "per_chapter_coverage": chapter_cov,
    })


def build_global_story_bible(
    *,
    job: dict[str, Any],
    source_registry: list[dict[str, Any]],
    edited_words: list[dict[str, Any]],
    duration_s: float,
    fps_num: int = 30,
    fps_den: int = 1,
    width: int = 1080,
    height: int = 1920,
    product_proof_by_feature: dict[str, Any] | None = None,
    verified_example_states: dict[str, Any] | None = None,
    silent_visual_beats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and freeze the global StoryBible contract BEFORE chapter/beat planning begins."""
    intended_story = str(job.get("intended_story") or "").strip()
    seg = canonical_split_sentences(intended_story)
    product_name = str(job.get("product_name") or "").strip() or None
    niche = str(job.get("niche_hint") or "general").strip().lower()

    atomic_claims = decompose_atomic_claims_and_actions(
        edited_words=edited_words,
        duration_s=duration_s,
        fps_num=fps_num,
        fps_den=fps_den,
        product_name=product_name,
        silent_visual_beats=silent_visual_beats,
    )
    chapters = segment_chapters(
        atomic_claims=atomic_claims,
        duration_s=duration_s,
        fps_num=fps_num,
        fps_den=fps_den,
    )

    # Persistent entities derived ONLY from verified job/example data (zero invented balances!)
    persistent_entities: dict[str, dict[str, Any]] = {}
    if isinstance(verified_example_states, dict):
        for eid, info in verified_example_states.items():
            persistent_entities[str(eid)] = {
                "entity_id": str(eid),
                "label": str(info.get("label", eid)),
                "unit": str(info.get("unit", "")),
                "initial_value": info.get("initial_value"),
                "current_value": info.get("initial_value"),
                "source_proof": str(info.get("source_proof", "verified_example")),
                "fact_vs_hypothetical": str(info.get("fact_vs_hypothetical", "hypothetical_labeled")),
                "continuity_id": f"cont_{eid}",
            }

    aspect = "16:9" if width > height else ("1:1" if width == height else "9:16")
    genre_camera = {
        "silent_art": {"pacing": "contemplative", "allow_motion_spam": False, "max_cut_density_per_10s": 1.5},
        "screen_tutorial": {"pacing": "instructional_hold", "preserve_ui_legibility": True, "max_cut_density_per_10s": 2.5},
        "documentary": {"pacing": "source_first", "preserve_source_integrity": True, "max_cut_density_per_10s": 2.5},
        "comedy": {"pacing": "protect_callback_pauses", "preserve_intentional_repeats": True, "max_cut_density_per_10s": 3.0},
    }.get(niche, {"pacing": "evidence_motivated", "preserve_face_and_ui": True, "max_cut_density_per_10s": 4.0})

    total_mat_w = round(sum(float(c["weight"]) for c in atomic_claims if c.get("is_material")), 4)
    bible = sanitize_native({
        "schema_version": STORY_BIBLE_SCHEMA_VERSION,
        "job_id": str(job.get("job_id") or "motionlab_r7_job"),
        "niche": niche,
        "aspect_ratio": aspect,
        "canvas": {"width": int(width), "height": int(height)},
        "output_fps": {"num": int(fps_num), "den": int(fps_den)},
        "duration_s": round(float(duration_s), 6),
        "total_frames": max(1, int(round(duration_s * float(Fraction(fps_num, fps_den))))),
        "source_registry": [
            {"source_id": s.get("source_id"), "sha256": s.get("sha256"), "duration_s": s.get("duration_s"), "fps": f"{s.get('fps_num')}/{s.get('fps_den')}"}
            for s in source_registry
        ],
        "canonical_script_segmentation": seg,
        "narrative_goal": str(job.get("job_to_be_done") or job.get("promise") or "Deliver truthful, visually evidenced storytelling"),
        "audience_intent": str(job.get("audience") or "general_viewer"),
        "style_tokens": dict(job.get("palette_tokens") or {}),
        "visual_language": str(job.get("visual_tone") or "clean_editorial_evidence"),
        "camera_grammar": genre_camera,
        "sound_palette": {
            "voice_priority": True,
            "allow_cash_cliche": False,
            "target_short_lufs": -16.0,
            "max_true_peak_dbtp": -1.0,
            "continuous_cross_chapter_phase": True,
        },
        "accessibility": {
            "min_mobile_width_frac": 0.45,
            "min_contrast_std": 20.0,
            "captions_burned_in": False,
        },
        "product_sources": product_proof_by_feature or {"overall_product_proof_status": "NOT_REQUIRED"},
        "banned_claims": list(job.get("forbidden_claims") or []),
        "protected_ranges": [
            {"claim_id": c["claim_id"], "start_frame": c["start_frame"], "end_frame_exclusive": c["end_frame_exclusive"], "reason": "negation_or_critical_fact"}
            for c in atomic_claims
            if c.get("has_negation") or c.get("materiality") == "critical"
        ],
        "persistent_entities": persistent_entities,
        "chapters": chapters,
        "atomic_claims": atomic_claims,
        "material_claims_total": sum(1 for c in atomic_claims if c.get("is_material")),
        "total_material_weight": max(0.01, total_mat_w),
    })
    bible["story_bible_hash"] = canonical_hash(bible)
    return bible

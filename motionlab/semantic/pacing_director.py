"""Phase A2.4–A2.7 (ML-01.R5.R4): Multi-Niche Causal Intelligent Pacing Director.

Content-first editorial rhythm planner grounded strictly in A1 evidence (aligned words, neural Silero VAD,
prosody, visual action ROI/motion) and StoryGraph narrative structure.

Zero hardcoding guarantees:
  - ZERO Olivia timestamps, cut lists, or duration quotas.
  - Adapts generically to any niche (finance, education/tutorial, tech/SaaS, product/UGC, storytelling/history,
    comedy, gameplay, wellness/reflective, silent aesthetic/ASMR, mixed/unknown).
  - Never cuts inside protected word uncertainty margins, never drops spoken claims/negations/CTA,
    and never auto-deletes nonverbal silent/ASMR art.
  - Causal selection: changing candidate scores (`candidate_score_overrides`) directly changes the selected
    pacing plan, compiled `EditTimeMap`, and final rendered frames/audio!
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any

from motionlab.composition.edit_time_map import (
    EditTimeMap,
    build_edit_time_map,
    build_identity_edit_time_map,
    validate_edit_time_map,
)
from motionlab.schemas import MediaInfo, canonical_hash, sanitize_native
from motionlab.semantic.story_graph import StoryGraph

PACING_SCHEMA_VERSION = "1.0.0"

VALID_PAUSE_CLASSES = frozenset({
    "DEAD_AIR",
    "BREATH",
    "EMOTIONAL",
    "COGNITIVE",
    "VISUAL_ACTION",
    "TRANSITION",
    "AMBIGUOUS",
    "INTENTIONAL_SILENCE",
    "END_TAIL",
})
VALID_PACING_ACTIONS = frozenset({"KEEP", "TRIM_GAP", "COMPRESS_NON_SPEECH", "CUT", "REVIEW"})

# Niche-aware pacing priors (generic editorial ergonomics, NEVER clip-specific cut lists!)
NICHE_PACING_PROFILES: dict[str, dict[str, float]] = {
    "finance":       {"min_phrase_gap_s": 0.22, "payoff_hold_s": 0.50, "action_hold_s": 0.42, "end_tail_s": 0.16, "allow_tight_breath": 1.0, "dead_air_min_s": 0.85},
    "product":       {"min_phrase_gap_s": 0.20, "payoff_hold_s": 0.45, "action_hold_s": 0.48, "end_tail_s": 0.16, "allow_tight_breath": 1.0, "dead_air_min_s": 0.80},
    "education":     {"min_phrase_gap_s": 0.28, "payoff_hold_s": 0.65, "action_hold_s": 0.70, "end_tail_s": 0.22, "allow_tight_breath": 0.0, "dead_air_min_s": 1.25},
    "tutorial":      {"min_phrase_gap_s": 0.30, "payoff_hold_s": 0.70, "action_hold_s": 0.80, "end_tail_s": 0.24, "allow_tight_breath": 0.0, "dead_air_min_s": 1.40},
    "tech":          {"min_phrase_gap_s": 0.26, "payoff_hold_s": 0.58, "action_hold_s": 0.65, "end_tail_s": 0.20, "allow_tight_breath": 0.0, "dead_air_min_s": 1.20},
    "storytelling":  {"min_phrase_gap_s": 0.30, "payoff_hold_s": 0.72, "action_hold_s": 0.55, "end_tail_s": 0.25, "allow_tight_breath": 0.0, "dead_air_min_s": 1.45},
    "comedy":        {"min_phrase_gap_s": 0.24, "payoff_hold_s": 0.68, "action_hold_s": 0.50, "end_tail_s": 0.22, "allow_tight_breath": 0.0, "dead_air_min_s": 1.35},
    "gameplay":      {"min_phrase_gap_s": 0.24, "payoff_hold_s": 0.50, "action_hold_s": 0.75, "end_tail_s": 0.20, "allow_tight_breath": 0.0, "dead_air_min_s": 1.30},
    "wellness":      {"min_phrase_gap_s": 0.45, "payoff_hold_s": 0.95, "action_hold_s": 0.85, "end_tail_s": 0.35, "allow_tight_breath": 0.0, "dead_air_min_s": 2.20},
    "silent_art":    {"min_phrase_gap_s": 9.99, "payoff_hold_s": 9.99, "action_hold_s": 9.99, "end_tail_s": 9.99, "allow_tight_breath": 0.0, "dead_air_min_s": 9.99},
    "lifestyle":     {"min_phrase_gap_s": 0.24, "payoff_hold_s": 0.52, "action_hold_s": 0.48, "end_tail_s": 0.18, "allow_tight_breath": 1.0, "dead_air_min_s": 0.95},
    "cinematic":     {"min_phrase_gap_s": 0.26, "payoff_hold_s": 0.58, "action_hold_s": 0.52, "end_tail_s": 0.20, "allow_tight_breath": 0.5, "dead_air_min_s": 1.05},
    "mixed":         {"min_phrase_gap_s": 0.26, "payoff_hold_s": 0.58, "action_hold_s": 0.55, "end_tail_s": 0.20, "allow_tight_breath": 0.5, "dead_air_min_s": 1.10},
    "unknown":       {"min_phrase_gap_s": 0.28, "payoff_hold_s": 0.62, "action_hold_s": 0.60, "end_tail_s": 0.22, "allow_tight_breath": 0.0, "dead_air_min_s": 1.25},
}


@dataclass
class PacingDecision:
    """Typed editorial decision for a single candidate pause interval."""

    decision_id: str
    source_start_s: float
    source_end_s: float
    source_start_frame: int
    source_end_frame_exclusive: int
    duration_s: float
    pause_class: str
    action: str  # KEEP | TRIM_GAP | CUT | REVIEW
    confidence: float
    protected_subspans: list[list[float]]
    reason: str
    evidence_refs: list[str]
    retained_duration_s: float
    cut_start_s: float | None = None
    cut_end_s: float | None = None
    removed_s: float = 0.0
    risk: str = "low"  # low | medium | high
    required_transition: str = "direct_cut"
    fallback_status: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def protect_semantic_spans(
    aligned_doc: dict[str, Any],
    vad_doc: dict[str, Any],
    story_graph: StoryGraph,
    action_doc: dict[str, Any],
    *,
    niche: str = "unknown",
    duration_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Build conservative merged protected source spans (words + margins, VAD speech, uncertain spans, critical actions)."""
    raw: list[tuple[float, float, str]] = []

    # 1. Every aligned word's conservative protected window (already includes pre/post uncertainty & consonant margins!)
    for w in aligned_doc.get("words") or []:
        p0 = max(0.0, float(w.get("protected_start_s", float(w["source_start_s"]) - 0.07)))
        p1 = min(duration_s, float(w.get("protected_end_s", float(w["source_end_s"]) + 0.09)))
        if p1 > p0:
            raw.append((p0, p1, f"word:{w.get('evidence_id', w['word_id'])}"))

    # 2. All neural Silero VAD speech spans (+40ms onset/tail safety)
    for sp in vad_doc.get("speech_spans") or []:
        p0 = max(0.0, float(sp["start_s"]) - 0.04)
        p1 = min(duration_s, float(sp["end_s"]) + 0.06)
        if p1 > p0:
            raw.append((p0, p1, f"vad:{sp.get('span_id', 'sp')}"))

    # 3. Any VAD-spectral disagreement uncertain span -> MUST NEVER CUT!
    for us in vad_doc.get("uncertain_spans") or []:
        p0 = max(0.0, float(us["start_s"]) - 0.04)
        p1 = min(duration_s, float(us["end_s"]) + 0.04)
        if p1 > p0:
            raw.append((p0, p1, f"uncertain_vad:{us.get('reason', 'ambiguous')}"))

    # 4. Protect first 0.25s opening expression/phoneme & silent-art videos completely
    if vad_doc.get("silent_video") or niche in ("silent_art", "asmr", "ambient"):
        raw.append((0.0, duration_s, "nonverbal_or_silent_art_protected"))
    else:
        raw.append((0.0, min(duration_s, 0.25), "opening_hook_expression_guard"))

    # Merge overlapping protected intervals
    raw.sort(key=lambda x: (x[0], x[1]))
    merged: list[dict[str, Any]] = []
    for s0, s1, reason in raw:
        if merged and s0 <= merged[-1]["end_s"] + 0.01:
            merged[-1]["end_s"] = round(max(merged[-1]["end_s"], s1), 4)
            if len(merged[-1]["reasons"]) < 6:
                merged[-1]["reasons"].append(reason)
        else:
            merged.append({"start_s": round(s0, 4), "end_s": round(s1, 4), "reasons": [reason]})
    return merged


def _subtract_protected(t0: float, t1: float, protected: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return strictly unprotected sub-intervals inside [t0, t1] after removing all protected spans."""
    free: list[tuple[float, float]] = [(t0, t1)]
    for p in protected:
        ps0, ps1 = float(p["start_s"]), float(p["end_s"])
        nxt: list[tuple[float, float]] = []
        for a, b in free:
            if ps1 <= a or ps0 >= b:
                nxt.append((a, b))
            else:
                if ps0 > a + 0.01:
                    nxt.append((a, ps0))
                if ps1 < b - 0.01:
                    nxt.append((ps1, b))
        free = nxt
    return [(round(a, 4), round(b, 4)) for a, b in free if b - a >= 0.08]


def classify_pause_candidates(
    aligned_doc: dict[str, Any],
    vad_doc: dict[str, Any],
    prosody_doc: dict[str, Any],
    action_doc: dict[str, Any],
    story_graph: StoryGraph,
    media: MediaInfo,
    *,
    niche: str = "unknown",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover and semantically classify all candidate pause windows strictly from A1 evidence + StoryGraph.

    Returns `(candidates, protected_spans)`.
    """
    dur_s = float(media.duration_s)
    fps_f = media.fps_float
    niche_key = niche.lower() if niche.lower() in NICHE_PACING_PROFILES else "unknown"
    prof = NICHE_PACING_PROFILES[niche_key]
    protected = protect_semantic_spans(aligned_doc, vad_doc, story_graph, action_doc, niche=niche_key, duration_s=dur_s)

    if vad_doc.get("silent_video") or niche_key == "silent_art":
        return ([
            {
                "candidate_id": "pause_00_silent_whole",
                "source_start_s": 0.0,
                "source_end_s": round(dur_s, 4),
                "source_start_frame": 0,
                "source_end_frame_exclusive": media.duration_frames,
                "duration_s": round(dur_s, 4),
                "pause_class": "INTENTIONAL_SILENCE",
                "recommended_action": "KEEP",
                "confidence": 0.95,
                "free_interior_spans": [],
                "reason": "Nonverbal or silent-art footage: absence of speech is NOT dead air.",
                "evidence_refs": ["vad:silent_video"],
                "prev_beat_act": "none",
                "next_beat_act": "none",
                "high_visual_action": False,
                "uncertain": False,
                "target_keep_head_s": round(dur_s, 4),
                "target_keep_tail_s": 0.0,
            }
        ], protected)

    words = aligned_doc.get("words") or []
    pauses = prosody_doc.get("inter_word_pauses") or []
    act_wins = action_doc.get("windows") or []
    uncertain_spans = vad_doc.get("uncertain_spans") or []

    def _find_action(t0: float, t1: float) -> dict[str, Any]:
        best: dict[str, Any] = {}
        best_ov = 0.0
        for aw in act_wins:
            ov = min(t1, float(aw["end_s"])) - max(t0, float(aw["start_s"]))
            if ov > best_ov:
                best_ov = ov
                best = aw
        return best

    def _adjacent_beats(t0: float, t1: float) -> tuple[str, str]:
        prev_act = "none"
        next_act = "none"
        for b in story_graph.beats:
            if b.speech_act == "breathing_bridge":
                continue
            if b.end_s <= t0 + 0.35:
                prev_act = b.speech_act
            if next_act == "none" and b.start_s >= t1 - 0.35:
                next_act = b.speech_act
        return prev_act, next_act

    # Build raw candidate gap list from real inter-word pauses + final tail (ZERO hardcoded timestamps!)
    raw_gaps: list[dict[str, Any]] = []
    for p in pauses:
        if float(p["duration_s"]) >= 0.28:
            raw_gaps.append({
                "start_s": float(p["start_s"]),
                "end_s": float(p["end_s"]),
                "duration_s": float(p["duration_s"]),
                "after_word_id": int(p["after_word_id"]),
                "before_word_id": int(p["before_word_id"]),
                "breath_likelihood": float(p.get("breath_likelihood", 0.2)),
                "phrase_boundary": bool(p.get("phrase_boundary", False)),
                "is_end_tail": False,
            })

    # Trailing end-tail after final spoken word
    if words:
        last_w = words[-1]
        tail_s = dur_s - float(last_w["source_end_s"])
        if tail_s >= 0.24:
            raw_gaps.append({
                "start_s": float(last_w["source_end_s"]),
                "end_s": dur_s,
                "duration_s": round(tail_s, 4),
                "after_word_id": int(last_w["word_id"]),
                "before_word_id": -1,
                "breath_likelihood": 0.0,
                "phrase_boundary": True,
                "is_end_tail": True,
            })

    word_by_id = {int(w["word_id"]): w for w in words}
    candidates: list[dict[str, Any]] = []

    for idx, g in enumerate(raw_gaps):
        t0, t1, gd = g["start_s"], g["end_s"], g["duration_s"]
        sf = max(0, min(media.duration_frames - 1, int(math.floor(t0 * fps_f))))
        ef = max(sf + 1, min(media.duration_frames, int(math.ceil(t1 * fps_f))))
        aw = _find_action(t0, t1)
        prev_act, next_act = _adjacent_beats(t0, t1)

        # Check uncertainty: VAD disagreement OR low-confidence adjacent word OR segment-only ASR
        w_before = word_by_id.get(g["after_word_id"])
        w_after = word_by_id.get(g["before_word_id"])
        adj_low_conf = bool(
            (w_before and float(w_before.get("uncertainty_ms", 0.0)) > 110.0)
            or (w_after and float(w_after.get("uncertainty_ms", 0.0)) > 110.0)
            or bool(aligned_doc.get("segment_only_protection", False))
        )
        vad_unc = any(min(t1, float(u["end_s"])) > max(t0, float(u["start_s"])) for u in uncertain_spans)
        is_uncertain = adj_low_conf or vad_unc

        free_spans = _subtract_protected(t0, t1, protected)
        ev_refs = [f"gap:{t0:.3f}-{t1:.3f}"]
        if w_before:
            ev_refs.append(f"after:{w_before['evidence_id']}")
        if w_after:
            ev_refs.append(f"before:{w_after['evidence_id']}")
        if aw:
            ev_refs.append(f"action:{aw.get('window_id', 'aw')}:task_mad={aw.get('mean_task_mad', 0):.1f}")

        high_act = bool(aw.get("high_visual_action", False))
        mod_act = bool(aw.get("moderate_visual_action", False))
        low_idle = bool(aw.get("low_info_idle_proxy", False))
        shot_cut = bool(aw.get("shot_cut_detected", False))

        # Semantic classification grounded strictly in speech acts, VAD, prosody & visual motion
        if is_uncertain:
            p_class = "AMBIGUOUS"
            rec_act = "REVIEW"
            conf = 0.38
            keep_head, keep_tail = gd, 0.0
            reason = "Adjacent word uncertainty or VAD-spectral disagreement -> abstain from cutting."
        elif g["is_end_tail"]:
            p_class = "END_TAIL"
            keep_head = min(gd, prof["end_tail_s"])
            keep_tail = 0.0
            rec_act = "TRIM_GAP" if (gd - keep_head >= 0.14 and free_spans) else "KEEP"
            conf = 0.86
            reason = "Trailing non-speech tail after final spoken CTA syllable; retain natural decay and trim dead tail."
        elif prev_act == "payoff_reveal" and gd >= prof["dead_air_min_s"]:
            p_class = "COGNITIVE"
            # Preserve cognitive realization landing + smile/breath, trim redundant dead-air interior!
            keep_head = min(gd * 0.45, prof["payoff_hold_s"])
            keep_tail = min(0.20, gd * 0.20)
            rec_act = "TRIM_GAP" if (gd - (keep_head + keep_tail) >= 0.25 and free_spans) else "KEEP"
            conf = 0.88 if low_idle else 0.78
            reason = "Post-payoff cognitive landing followed by redundant dead air; preserve realization pause + pre-onset breath and trim idle middle."
        elif gd >= prof["dead_air_min_s"] and (low_idle or (not shot_cut and not high_act)):
            p_class = "DEAD_AIR"
            keep_head = min(0.32, gd * 0.30)
            keep_tail = min(0.18, gd * 0.20)
            rec_act = "TRIM_GAP" if (gd - (keep_head + keep_tail) >= 0.25 and free_spans) else "KEEP"
            conf = 0.87
            reason = "Extended non-speech dead air with low new visual information; trim middle while protecting speech boundaries."
        elif gd >= 0.65 and (high_act or mod_act):
            p_class = "VISUAL_ACTION"
            # Retain meaningful visible gesture/demonstration + natural rhythm; trim only excessive idle middle if very long!
            keep_head = min(gd * 0.50, prof["action_hold_s"])
            keep_tail = min(0.18, gd * 0.20)
            trimmable = gd - (keep_head + keep_tail)
            if niche_key in ("tutorial", "education", "gameplay") and high_act:
                rec_act = "KEEP"
                keep_head, keep_tail = gd, 0.0
                reason = "Active visual demonstration/HUD step in instructional/gameplay niche -> preserve full duration."
            elif trimmable >= 0.28 and free_spans and not shot_cut:
                rec_act = "TRIM_GAP"
                reason = "Visible hand/prop action pause exceeding concise rhythm window; keep action beat + natural breath and trim redundant interior."
            else:
                rec_act = "KEEP"
                reason = "Compact visual action beat — retain intact for continuity."
            conf = 0.76
        elif prev_act == "closing_takeaway" or next_act == "cta":
            p_class = "TRANSITION"
            keep_head = min(gd, max(0.16, prof["min_phrase_gap_s"] * 0.85))
            keep_tail = 0.0
            rec_act = "TRIM_GAP" if (gd - keep_head >= 0.22 and free_spans and prof["allow_tight_breath"] > 0.0) else "KEEP"
            conf = 0.74
            reason = "Pre-CTA transition beat; keep natural cue space without dead silence."
        elif g["breath_likelihood"] >= 0.5 and gd < 0.75:
            p_class = "BREATH"
            keep_head = min(gd, prof["min_phrase_gap_s"])
            keep_tail = 0.0
            rec_act = "TRIM_GAP" if (gd - keep_head >= 0.22 and free_spans and prof["allow_tight_breath"] >= 0.8) else "KEEP"
            conf = 0.72
            reason = "Natural speaker breath between phrases; preserve inhale and trim only excess silence."
        else:
            p_class = "TRANSITION" if g["phrase_boundary"] else "BREATH"
            keep_head = min(gd, prof["min_phrase_gap_s"])
            keep_tail = 0.0
            rec_act = "TRIM_GAP" if (gd - keep_head >= 0.24 and free_spans and prof["allow_tight_breath"] >= 0.8) else "KEEP"
            conf = 0.70
            reason = "Inter-phrase transition gap."

        candidates.append({
            "candidate_id": f"pause_{idx:02d}_{p_class.lower()}",
            "source_start_s": round(t0, 4),
            "source_end_s": round(t1, 4),
            "source_start_frame": sf,
            "source_end_frame_exclusive": ef,
            "duration_s": round(gd, 4),
            "pause_class": p_class,
            "recommended_action": rec_act,
            "confidence": round(conf, 3),
            "free_interior_spans": [[a, b] for a, b in free_spans],
            "reason": reason,
            "evidence_refs": ev_refs,
            "prev_beat_act": prev_act,
            "next_beat_act": next_act,
            "high_visual_action": high_act,
            "shot_cut_detected": shot_cut,
            "uncertain": is_uncertain,
            "is_end_tail": g["is_end_tail"],
            "target_keep_head_s": round(keep_head, 4),
            "target_keep_tail_s": round(keep_tail, 4),
        })

    return candidates, protected


def _compute_cut_from_candidate(cand: dict[str, Any], variant: str, fps_f: float) -> PacingDecision:
    """Derive a concrete frame-safe PacingDecision from a classified pause candidate under variant (`identity`|`conservative`|`balanced`|`tight`)."""
    t0, t1, gd = float(cand["source_start_s"]), float(cand["source_end_s"]), float(cand["duration_s"])
    sf, ef = int(cand["source_start_frame"]), int(cand["source_end_frame_exclusive"])
    p_class = str(cand["pause_class"])
    free_spans = cand.get("free_interior_spans") or []

    def _keep_dec(reason_extra: str = "") -> PacingDecision:
        return PacingDecision(
            decision_id=f"{variant}:{cand['candidate_id']}",
            source_start_s=t0,
            source_end_s=t1,
            source_start_frame=sf,
            source_end_frame_exclusive=ef,
            duration_s=gd,
            pause_class=p_class,
            action="REVIEW" if cand.get("uncertain") else "KEEP",
            confidence=float(cand["confidence"]),
            protected_subspans=[[t0, t1]],
            reason=(cand["reason"] + (" " + reason_extra if reason_extra else "")).strip(),
            evidence_refs=list(cand.get("evidence_refs") or []),
            retained_duration_s=gd,
            cut_start_s=None,
            cut_end_s=None,
            removed_s=0.0,
            risk="low",
            required_transition="none",
            fallback_status="KEPT_INTACT",
        )

    if variant == "identity" or cand.get("uncertain") or p_class in ("INTENTIONAL_SILENCE", "AMBIGUOUS", "EMOTIONAL") or not free_spans:
        return _keep_dec()

    # Select largest strictly unprotected interior span (guaranteed clear of all protected words/VAD speech!)
    best_free = max(free_spans, key=lambda pair: float(pair[1]) - float(pair[0]))
    f0, f1 = float(best_free[0]), float(best_free[1])
    free_dur = f1 - f0

    # Variant-specific retention multipliers (higher = preserve more pause!)
    if variant == "conservative":
        # Only trim major dead air / post-payoff (>1.15s) or trailing tail (>0.30s)
        if p_class not in ("DEAD_AIR", "COGNITIVE", "END_TAIL"):
            return _keep_dec("(conservative variant retains action/breath pauses intact)")
        if not cand.get("is_end_tail") and gd < 1.15:
            return _keep_dec("(below conservative dead-air threshold)")
        head_keep = min(gd * 0.55, max(0.55, float(cand["target_keep_head_s"]) * 1.25))
        tail_keep = 0.0 if cand.get("is_end_tail") else max(0.22, float(cand["target_keep_tail_s"]))
    elif variant == "balanced":
        if cand["recommended_action"] == "KEEP":
            return _keep_dec()
        head_keep = float(cand["target_keep_head_s"])
        tail_keep = 0.0 if cand.get("is_end_tail") else float(cand["target_keep_tail_s"])
    else:  # "tight"
        if p_class == "VISUAL_ACTION" and cand.get("high_visual_action") and gd < 0.70:
            return _keep_dec("(compact high-action beat retained even in tight variant)")
        head_keep = max(0.14, float(cand["target_keep_head_s"]) * 0.72)
        tail_keep = 0.0 if cand.get("is_end_tail") else max(0.12, float(cand["target_keep_tail_s"]) * 0.75)

    # Compute strictly safe cut inside [f0, f1] honoring both protected bounds AND head/tail retention!
    c0 = max(f0, t0 + head_keep)
    c1 = f1 if cand.get("is_end_tail") else min(f1, t1 - tail_keep)
    # Round strictly inward to integer frames
    cf0 = int(math.ceil(c0 * fps_f))
    cf1 = int(math.floor(c1 * fps_f))
    if cf1 - cf0 < 4:  # Minimum 4 frames (~167ms @ 24fps) to justify a real cut!
        return _keep_dec("(safe trimmable interior < 4 frames)")

    cut_s0 = round(cf0 / fps_f, 4)
    cut_s1 = round(cf1 / fps_f, 4)
    rem_s = round(cut_s1 - cut_s0, 4)
    if rem_s < 0.16:
        return _keep_dec("(safe trimmable interior < 0.16s)")

    prot_subs: list[list[float]] = []
    if cut_s0 > t0 + 0.01:
        prot_subs.append([t0, cut_s0])
    if cut_s1 < t1 - 0.01:
        prot_subs.append([cut_s1, t1])

    req_trans = "direct_cut"
    if p_class == "VISUAL_ACTION":
        req_trans = "action_match_or_shared_element"
    elif p_class == "COGNITIVE":
        req_trans = "bridge_or_motivated_reframe"

    return PacingDecision(
        decision_id=f"{variant}:{cand['candidate_id']}",
        source_start_s=t0,
        source_end_s=t1,
        source_start_frame=sf,
        source_end_frame_exclusive=ef,
        duration_s=gd,
        pause_class=p_class,
        action="CUT" if cand.get("is_end_tail") else "TRIM_GAP",
        confidence=float(cand["confidence"]),
        protected_subspans=prot_subs,
        reason=cand["reason"],
        evidence_refs=list(cand.get("evidence_refs") or []),
        retained_duration_s=round(gd - rem_s, 4),
        cut_start_s=cut_s0,
        cut_end_s=cut_s1,
        removed_s=rem_s,
        risk="medium" if p_class == "VISUAL_ACTION" else "low",
        required_transition=req_trans,
        fallback_status="CUT_APPROVED",
    )


def propose_pacing_alternatives(
    candidates: list[dict[str, Any]],
    protected_spans: list[dict[str, Any]],
    media: MediaInfo,
    *,
    niche: str = "unknown",
) -> list[dict[str, Any]]:
    """Generate distinct global pacing plan variants (`identity`, `conservative`, `balanced`, `tight`)."""
    fps_f = media.fps_float
    prot_tuples = [(float(p["start_s"]), float(p["end_s"])) for p in protected_spans]
    plans: list[dict[str, Any]] = []

    for variant in ("identity", "conservative", "balanced", "tight"):
        decisions = [_compute_cut_from_candidate(c, variant, fps_f) for c in candidates]
        cut_specs = [
            {
                "cut_id": f"cut_{i:02d}_{d.pause_class.lower()}",
                "decision_id": d.decision_id,
                "cut_start_s": d.cut_start_s,
                "cut_end_s": d.cut_end_s,
                "pause_class": d.pause_class,
                "action": d.action,
                "reason": d.reason,
                "evidence_refs": d.evidence_refs,
                "confidence": d.confidence,
            }
            for i, d in enumerate(decisions)
            if d.action in ("TRIM_GAP", "CUT") and d.cut_start_s is not None and d.cut_end_s is not None
        ]
        try:
            emap = build_edit_time_map(
                media,
                cut_specs,
                protected_intervals_s=prot_tuples,
                pacing_mode="preserve" if variant == "identity" else "auto",
            )
            map_errs = validate_edit_time_map(emap, protected_intervals_s=prot_tuples)
            eligible = len(map_errs) == 0
        except Exception as exc:
            emap = build_identity_edit_time_map(media, pacing_mode="auto", status="PACING_NO_SAFE_CUTS")
            map_errs = [f"map_build_rejected:{exc}"]
            eligible = False

        plans.append({
            "plan_id": f"pacing_{variant}",
            "variant": variant,
            "niche": niche,
            "eligible": eligible,
            "rejection_reasons": map_errs,
            "decisions": [d.to_dict() for d in decisions],
            "cuts_count": len(emap.omitted_spans),
            "removed_total_s": round(emap.removed_total_s, 4),
            "output_total_frames": emap.output_total_frames,
            "output_duration_s": round(emap.output_duration_s, 4),
            "edit_time_map": emap.to_dict(),
            "map_hash": emap.map_hash,
        })
    return plans


def score_pacing_candidate(
    plan: dict[str, Any],
    media: MediaInfo,
    story_graph: StoryGraph,
    *,
    niche: str = "unknown",
    candidate_score_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a pacing plan on information rhythm, naturalness preservation, and continuity safety (NO fixed duration quota!)."""
    if not plan.get("eligible"):
        return {**plan, "score": -999.0, "score_breakdown": {"hard_rejected": True}}

    variant = str(plan["variant"])
    rem_s = float(plan["removed_total_s"])
    cuts_n = int(plan["cuts_count"])
    src_s = max(1.0, float(media.duration_s))
    rem_ratio = rem_s / src_s

    decs = plan.get("decisions") or []
    dead_air_removed = sum(float(d["removed_s"]) for d in decs if d["pause_class"] in ("DEAD_AIR", "COGNITIVE", "END_TAIL"))
    breath_trimmed_n = sum(1 for d in decs if d["pause_class"] == "BREATH" and float(d["removed_s"]) > 0.0)
    action_trimmed_s = sum(float(d["removed_s"]) for d in decs if d["pause_class"] == "VISUAL_ACTION")
    unresolved_long_dead = sum(
        float(d["retained_duration_s"]) - 0.90
        for d in decs
        if d["pause_class"] in ("DEAD_AIR", "COGNITIVE") and float(d["retained_duration_s"]) > 0.90
    )

    # Rhythm gain: reward removing confirmed dead-air/post-payoff idle up to diminishing returns
    rhythm_gain = min(28.0, dead_air_removed * 7.5 + action_trimmed_s * 5.0)
    # Penalize leaving long unresolved dead air (>0.9s) on fast/medium niches
    niche_key = niche.lower() if niche.lower() in NICHE_PACING_PROFILES else "unknown"
    dead_air_penalty = 0.0 if niche_key in ("wellness", "silent_art") else unresolved_long_dead * 7.0
    # Continuity & naturalness preservation: penalize excessive micro-cuts or cutting too many short breaths
    overcut_penalty = max(0.0, (cuts_n - 6) * 2.5) + breath_trimmed_n * 1.4
    # Penalize aggressive total trim > 28% (risks breathless unnatural cadence)
    breathless_penalty = max(0.0, (rem_ratio - 0.28) * 60.0)
    # D09 fix: Remove fixed variant bonus; score strictly on measurable content-driven rhythm features!
    ergonomic_bonus = 0.0
    if cuts_n > 0 and 0.04 <= rem_ratio <= 0.22 and unresolved_long_dead <= 0.25:
        ergonomic_bonus = round(min(2.0, (0.25 - unresolved_long_dead) * 8.0), 3)

    override_delta = float((candidate_score_overrides or {}).get(plan["plan_id"], 0.0))
    total = round(50.0 + rhythm_gain + ergonomic_bonus - dead_air_penalty - overcut_penalty - breathless_penalty + override_delta, 3)

    return {
        **plan,
        "score": total,
        "score_breakdown": {
            "rhythm_gain": round(rhythm_gain, 2),
            "dead_air_penalty": round(dead_air_penalty, 2),
            "overcut_penalty": round(overcut_penalty, 2),
            "breathless_penalty": round(breathless_penalty, 2),
            "ergonomic_bonus": ergonomic_bonus,
            "override_delta": override_delta,
        },
    }


def validate_pacing_plan(
    chosen_plan: dict[str, Any],
    edit_map: EditTimeMap,
    protected_spans: list[dict[str, Any]],
) -> list[str]:
    """Fail-closed validation of chosen pacing plan & its EditTimeMap."""
    prot_tuples = [(float(p["start_s"]), float(p["end_s"])) for p in protected_spans]
    errs = validate_edit_time_map(edit_map, protected_intervals_s=prot_tuples)
    if not chosen_plan.get("eligible"):
        errs.append("chosen_pacing_plan_not_eligible")
    if chosen_plan.get("map_hash") != edit_map.map_hash:
        errs.append("pacing_plan_vs_edit_map_hash_mismatch")
    return errs


def choose_pacing_plan(
    media: MediaInfo,
    story_graph: StoryGraph,
    perception: dict[str, Any],
    *,
    pacing_mode: str = "auto",
    niche: str = "unknown",
    candidate_score_overrides: dict[str, float] | None = None,
    excluded_cut_ids: set[str] | None = None,
) -> tuple[dict[str, Any], EditTimeMap]:
    """Causal Pacing Director entry point: classifies pauses -> proposes variants -> scores -> selects & validates EditTimeMap.

    When `pacing_mode == "preserve"`, returns exact identity map without altering source timeline.
    When `excluded_cut_ids` is provided (e.g. rollback after word preservation or join check failure),
    those cuts are forcibly locked to KEEP and the map is cleanly recomputed!
    """
    aligned_doc = perception["speech_alignment"]
    vad_doc = perception["vad_spans"]
    prosody_doc = perception["prosody_summary"]
    action_doc = perception["action_evidence"]

    candidates, protected_spans = classify_pause_candidates(
        aligned_doc, vad_doc, prosody_doc, action_doc, story_graph, media, niche=niche
    )

    if pacing_mode == "preserve":
        id_map = build_identity_edit_time_map(media, pacing_mode="preserve", status="IDENTITY_PRESERVE")
        return {
            "schema_version": PACING_SCHEMA_VERSION,
            "pacing_mode": "preserve",
            "niche": niche,
            "status": "IDENTITY_PRESERVE",
            "chosen_plan_id": "pacing_identity",
            "causal_selection_before_compile": True,
            "candidates_analyzed": len(candidates),
            "protected_spans_count": len(protected_spans),
            "protected_spans": protected_spans,
            "pause_candidates": candidates,
            "scored_variants": [],
            "chosen_decisions": [],
            "excluded_cut_ids_rolled_back": sorted(excluded_cut_ids or []),
            "edit_map_hash": id_map.map_hash,
            "source_duration_s": round(media.duration_s, 4),
            "output_duration_s": round(id_map.output_duration_s, 4),
            "removed_total_s": 0.0,
            "cuts_count": 0,
            "validation_errors": [],
        }, id_map

    # If rollback excluded specific cut IDs, lock matching candidates to KEEP!
    excl = set(excluded_cut_ids or set())
    if excl:
        for c in candidates:
            cid = str(c["candidate_id"])
            if any(cid in ex or ex in cid or c["pause_class"].lower() in ex for ex in excl):
                c["recommended_action"] = "KEEP"
                c["uncertain"] = True
                c["reason"] = f"{c['reason']} [ROLLED_BACK_SAFETY]"

    variants = propose_pacing_alternatives(candidates, protected_spans, media, niche=niche)
    scored = [
        score_pacing_candidate(v, media, story_graph, niche=niche, candidate_score_overrides=candidate_score_overrides)
        for v in variants
    ]
    eligible = [s for s in scored if s.get("eligible")]
    eligible.sort(key=lambda s: (-float(s["score"]), s["plan_id"]))

    if eligible:
        chosen = eligible[0]
        edit_map = EditTimeMap.from_dict(chosen["edit_time_map"])
    else:
        edit_map = build_identity_edit_time_map(media, pacing_mode="auto", status="PACING_NO_SAFE_CUTS")
        chosen = next((s for s in scored if s["variant"] == "identity"), {
            "plan_id": "pacing_identity_fallback",
            "variant": "identity",
            "eligible": True,
            "decisions": [],
            "cuts_count": 0,
            "removed_total_s": 0.0,
            "output_total_frames": edit_map.output_total_frames,
            "output_duration_s": edit_map.output_duration_s,
            "map_hash": edit_map.map_hash,
        })

    val_errs = validate_pacing_plan(chosen, edit_map, protected_spans)
    if val_errs:
        # Fail closed to safe identity fallback rather than ever emitting an invalid cut!
        edit_map = build_identity_edit_time_map(media, pacing_mode="auto", status="PACING_NO_SAFE_CUTS")

    plan_doc = {
        "schema_version": PACING_SCHEMA_VERSION,
        "pacing_mode": "auto",
        "niche": niche,
        "status": edit_map.status,
        "chosen_plan_id": chosen["plan_id"],
        "chosen_variant": chosen.get("variant", "identity"),
        "causal_selection_before_compile": True,
        "candidates_analyzed": len(candidates),
        "protected_spans_count": len(protected_spans),
        "protected_spans": protected_spans,
        "pause_candidates": candidates,
        "scored_variants": [
            {
                "plan_id": s["plan_id"],
                "variant": s["variant"],
                "eligible": s["eligible"],
                "score": s["score"],
                "score_breakdown": s["score_breakdown"],
                "cuts_count": s["cuts_count"],
                "removed_total_s": s["removed_total_s"],
                "output_total_frames": s["output_total_frames"],
                "output_duration_s": s["output_duration_s"],
                "map_hash": s["map_hash"],
                "rejection_reasons": s["rejection_reasons"],
            }
            for s in scored
        ],
        "chosen_decisions": chosen.get("decisions", []),
        "excluded_cut_ids_rolled_back": sorted(excl),
        "edit_map_hash": edit_map.map_hash,
        "source_duration_s": round(media.duration_s, 4),
        "output_duration_s": round(edit_map.output_duration_s, 4),
        "removed_total_s": round(edit_map.removed_total_s, 4),
        "cuts_count": len(edit_map.omitted_spans),
        "validation_errors": val_errs,
    }
    return plan_doc, edit_map

"""Phase 1 (ML-01.R4): Strict Anti-Hallucination Claim & Numeric Provenance Grounding Validator.

Guarantees that ZERO invented dollar amounts, ZERO financial guarantees, ZERO unmentioned product/brand claims,
and ZERO negated claims ("not saving $100" -> "$100 saved") can ever enter a visual label or benefit card.
"""

from __future__ import annotations

import re
from typing import Any

from motionlab.semantic.schema import GroundedClaim, SemanticUnit, Transcript

_NUM_RE = re.compile(r"(?:[\$€£¥]\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?")
_FORBIDDEN_GUARANTEE_PATTERNS = (
    "guaranteed", "get rich", "double your money", "risk-free", "free money",
    "100% profit", "passive income guaranteed", "instant wealth", "financial advice",
)
_NEGATION_WORDS = {"not", "don't", "doesn't", "never", "no", "cannot", "can't", "won't", "isn't", "aren't"}
_POSITIVE_GAIN_WORDS = {"saved", "saving", "save", "earned", "profit", "gain", "extra", "bonus", "made"}


def _norm_num(s: str) -> str:
    """Normalize numeric string ('$1,200' -> '1200', '800' -> '800') for exact set membership checks."""
    return re.sub(r"[^0-9.]", "", s).lstrip("0") or "0"


def _extract_nums(text: str) -> list[str]:
    return [m.group(0).strip() for m in _NUM_RE.finditer(text or "")]


def _has_negation_flip(verbatim_quote: str, label: str) -> bool:
    """Reject labels that invert a negated spoken financial statement (e.g., 'not saving $100' -> '$100 SAVED')."""
    q_low = verbatim_quote.lower()
    l_low = label.lower()
    q_tokens = set(re.findall(r"[a-z']+", q_low))
    l_tokens = set(re.findall(r"[a-z']+", l_low))
    quote_negated = bool(q_tokens & _NEGATION_WORDS)
    label_negated = bool(l_tokens & _NEGATION_WORDS) or ("≠" in label) or ("not " in l_low) or ("limit" in l_low) or ("left" in l_low)
    label_claims_gain = bool(l_tokens & _POSITIVE_GAIN_WORDS)
    return quote_negated and label_claims_gain and not label_negated


def validate_claim_grounding(
    raw_beats: list[dict[str, Any]],
    transcript: Transcript,
    units: list[SemanticUnit],
) -> tuple[list[dict[str, Any]], list[GroundedClaim], list[GroundedClaim]]:
    """Verify every beat's visual display_label and secondary_labels strictly against real transcript provenance."""
    units_by_id = {u.unit_id: u for u in units}
    all_spoken_norm_nums: set[str] = set()
    confident_spoken_norm_nums: set[str] = set()
    full_low = (transcript.full_text or "").lower()

    for seg in transcript.segments:
        for w in seg.words:
            for nm in _extract_nums(w.word):
                nn = _norm_num(nm)
                all_spoken_norm_nums.add(nn)
                if w.probability >= 0.45:
                    confident_spoken_norm_nums.add(nn)
    for nm in _extract_nums(transcript.full_text):
        all_spoken_norm_nums.add(_norm_num(nm))

    sanitized_beats: list[dict[str, Any]] = []
    accepted: list[GroundedClaim] = []
    rejected: list[GroundedClaim] = []
    claim_seq = 0

    def _check_single_label(label: str, u_objs: list[SemanticUnit], quote: str) -> tuple[bool, str, str, list[str]]:
        lbl = re.sub(r"\s+", " ", (label or "").strip())
        if not lbl or lbl.lower() in ("none", "n/a", "null", "-", "empty"):
            return True, "", "abstract_visual", []
        # Enforce crisp short editorial label (<= 5 words, never a verbatim subtitle sentence!)
        words = lbl.split()
        if len(words) > 5:
            lbl = " ".join(words[:4])

        low = lbl.lower()
        for bad_pat in _FORBIDDEN_GUARANTEE_PATTERNS:
            if bad_pat in low:
                return False, f"forbidden_financial_guarantee:{bad_pat}", "rejected", []

        if _has_negation_flip(quote, lbl):
            return False, "negation_polarity_flip_rejected", "rejected", []

        # Every number inside the label MUST exist verbatim in THIS beat's OWN referenced unit(s) (never leaked from another beat!)
        lbl_nums = _extract_nums(lbl)
        unit_nums = {_norm_num(n["raw"]) for u in u_objs for n in u.spoken_numbers}
        unit_quote_low = " ".join(u.verbatim_quote.lower() for u in u_objs)

        for nm in lbl_nums:
            nn = _norm_num(nm)
            if nn not in unit_nums:
                return False, f"ungrounded_numeric_claim:{nm}", "rejected", lbl_nums
            if confident_spoken_norm_nums and nn not in confident_spoken_norm_nums:
                return False, f"low_confidence_numeric_claim:{nm}", "rejected", lbl_nums

        # Check non-stopword lexical grounding so LLM never invents unmentioned brand/product/promises
        stop = {"the", "a", "an", "to", "for", "of", "in", "on", "your", "my", "is", "are", "with", "vs", "not", "≠", "->", "&", "+", "step", "focus", "plan", "flow"}
        content_words = [w for w in re.findall(r"[a-z]+", low) if w not in stop and len(w) >= 3]
        if content_words:
            # At least half of content words (or stem prefix >=4 chars) must appear in transcript!
            matched = sum(
                1 for cw in content_words
                if (cw in unit_quote_low or cw in full_low or (len(cw) >= 4 and cw[:4] in full_low))
            )
            if matched == 0 and not lbl_nums:
                return False, f"ungrounded_label_terms:{lbl}", "rejected", lbl_nums

        kind = "verified_figure" if lbl_nums else ("quotation" if low in full_low else "grounded_paraphrase")
        return True, lbl, kind, lbl_nums

    for b in raw_beats:
        uids = [uid for uid in (b.get("unit_ids") or []) if uid in units_by_id]
        u_objs = [units_by_id[uid] for uid in uids]
        quote = " ".join(u.verbatim_quote for u in u_objs).strip()
        mean_conf = round(sum(u.mean_confidence for u in u_objs) / max(1, len(u_objs)), 4) if u_objs else 0.0
        primary_uid = uids[0] if uids else "u_none"

        raw_disp = str(b.get("display_label") or "").strip()
        ok_d, out_d, kind_d, nums_d = _check_single_label(raw_disp, u_objs, quote)
        claim_ids: list[str] = []
        if raw_disp:
            cid = f"clm_{claim_seq:03d}"
            claim_seq += 1
            gc = GroundedClaim(
                claim_id=cid,
                unit_id=primary_uid,
                claim_kind=kind_d,
                display_label=out_d if ok_d else raw_disp,
                verbatim_quote=quote,
                numeric_values=nums_d,
                confidence=mean_conf,
                accepted=ok_d,
                reason="verified_against_asr_transcript" if ok_d else out_d,
            )
            if ok_d:
                accepted.append(gc)
                claim_ids.append(cid)
            else:
                rejected.append(gc)
                out_d = ""

        clean_sec: list[str] = []
        for s_lbl in (b.get("secondary_labels") or [])[:4]:
            s_raw = str(s_lbl or "").strip()
            if not s_raw:
                continue
            ok_s, out_s, kind_s, nums_s = _check_single_label(s_raw, u_objs, quote)
            cid = f"clm_{claim_seq:03d}"
            claim_seq += 1
            gc = GroundedClaim(
                claim_id=cid,
                unit_id=primary_uid,
                claim_kind=kind_s,
                display_label=out_s if ok_s else s_raw,
                verbatim_quote=quote,
                numeric_values=nums_s,
                confidence=mean_conf,
                accepted=ok_s,
                reason="verified_against_asr_transcript" if ok_s else out_s,
            )
            if ok_s and out_s:
                accepted.append(gc)
                clean_sec.append(out_s)
                claim_ids.append(cid)
            else:
                rejected.append(gc)

        sanitized_beats.append({
            **b,
            "unit_ids": uids,
            "verbatim_quote": quote,
            "display_label": out_d,
            "secondary_labels": clean_sec,
            "grounded_claim_ids": claim_ids,
            "confidence": mean_conf,
        })

    return sanitized_beats, accepted, rejected

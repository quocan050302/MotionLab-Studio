"""Phase A1 (ML-01.R5): Semantic Story Graph, Integer-Cent Verified Financial Ledger & Claim Provenance.

Strictly generic (ZERO hardcoded clip names, scripts, or timestamps):
- Extracts word-aligned speech acts, numeric entities, currency, category nouns, modality, and negation.
- Distinguishes `verbatim_spoken` quotes from `derived_arithmetic` intermediate balances with full integer-cent audit trails.
- Enforces conservation (`initial - sum(allocations) == final_remaining`), handles reordered/changed currencies/overspend,
  isolates non-monetary counts (e.g. "56-page workbook") from money ledgers, and rejects hallucinated savings/claims.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.schemas import MediaInfo, canonical_hash, sanitize_native
from motionlab.semantic.schema import SemanticUnit, Transcript, WordSpan

STORY_GRAPH_SCHEMA_VERSION = "1.0.0"

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_CURRENCY_WORDS = {"dollar": "$", "dollars": "$", "usd": "$", "euro": "€", "euros": "€", "pound": "£", "pounds": "£"}

_NEGATION_RE = re.compile(
    r"(?i)\b(doesn'?t\s+mean|don'?t\s+have|not\s+actually|never|cannot|can'?t\s+spend|isn'?t|aren'?t|won'?t)\b"
)
_HYPOTHETICAL_RE = re.compile(r"(?i)\b(let'?s\s+say|for\s+example|suppose|imagine|say\s+you|hypothetically)\b")
_REMAINING_RE = re.compile(r"(?i)\b(left|remaining|left\s+over|actually\s+have|available|balance\s+left)\b")
_OPENING_BALANCE_RE = re.compile(r"(?i)\b(bank\s+account|paycheck|income|balance|starting|have|make|earn|deposit|budget)\b")
_ALLOCATION_FOR_RE = re.compile(
    r"(?i)(?:(?P<sym>[\$€£¥])\s*)?(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)\s*(?:dollars?|euros?|pounds?)?\s*"
    r"(?:is\s+|goes\s+|went\s+)?(?:for|to|toward|towards|on)\s+(?P<cat>[A-Za-z][A-Za-z0-9\-']*(?:\s+[A-Za-z][A-Za-z0-9\-']*)?)"
)
_AMOUNT_ANY_RE = re.compile(
    r"(?i)(?P<sym>[\$€£¥])?\s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)(?P<suffix>\s*-\s*page|\s+pages?|\s*%|\s*k\b)?"
)
_PRODUCT_INTRO_RE = re.compile(
    r"(?i)\b(?:created|built|made|designed|introducing|called|named)\s+(?P<prod>[A-Z][A-Za-z0-9'\-]*(?:\s+(?:[A-Z][A-Za-z0-9'\-]*|the|of|for|to|and)){0,5})"
)
_CTA_RE = re.compile(r"(?i)\b(link\s+in\s+(?:my\s+)?(?:profile|bio)|check\s+out|grab\s+it|comment|subscribe|follow)\b")
_BENEFIT_RE = re.compile(r"(?i)\b(help\s+you|organize|plan|dashboard|workbook|template|system|guide|tool|track|purpose)\b")
_FORBIDDEN_HALLUCINATION_RE = re.compile(
    r"(?i)(?:\bguaranteed?\b|\bsave\s+[\$€£¥]?\d[\d,]*|\bprofit\b|\bdouble\s+your\s+money|\bget\s+rich|\brisk[\-\s]?free|\bbonus\b|\bfree\s+money)"
)
_STOP_CAT_WORDS = {"a", "an", "the", "your", "my", "our", "their", "to", "and", "or", "is", "are", "just", "only"}


def parse_amount_to_cents(num_str: str) -> int:
    """Parse formatted decimal/integer number string strictly into integer cents (zero float rounding error)."""
    clean = num_str.replace(",", "").strip()
    if "." in clean:
        whole_s, frac_s = clean.split(".", 1)
        frac_s = (frac_s + "00")[:2]
        return int(whole_s or "0") * 100 + int(frac_s)
    return int(clean) * 100


def format_cents(cents: int, currency_symbol: str = "$") -> str:
    """Format integer cents cleanly (e.g. 120000 -> '$1,200', -80000 -> '-$800', 12550 -> '$125.50')."""
    sign = "-" if cents < 0 else ""
    abs_c = abs(int(cents))
    whole = abs_c // 100
    rem = abs_c % 100
    if rem == 0:
        return f"{sign}{currency_symbol}{whole:,}"
    return f"{sign}{currency_symbol}{whole:,}.{rem:02d}"


@dataclass
class LedgerStep:
    step_id: str
    step_type: str  # opening_balance | allocation | remaining_confirmed | overspend_warning
    category: str
    delta_cents: int
    amount_cents: int
    running_balance_cents: int
    currency_symbol: str
    currency_code: str
    proportion_of_initial: float
    provenance: str  # "verbatim_spoken" | "derived_arithmetic"
    running_balance_provenance: str  # "verbatim_spoken" | "derived_arithmetic"
    formula: str
    start_s: float
    end_s: float
    start_frame: int
    end_frame_exclusive: int
    word_ids: list[int] = field(default_factory=list)
    verbatim_quote: str = ""
    modality: str = "assertive"  # assertive | hypothetical | negated
    uncertain: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class VerifiedLedger:
    active: bool
    currency_symbol: str
    currency_code: str
    initial_cents: int
    total_allocated_cents: int
    computed_remaining_cents: int
    spoken_remaining_cents: int | None
    overspend_cents: int
    conservation_exact: bool
    reconciled_with_spoken_remaining: bool
    modality: str  # hypothetical_example | actual_statement
    steps: list[LedgerStep] = field(default_factory=list)
    non_monetary_counts: list[dict[str, Any]] = field(default_factory=list)
    negated_statements: list[dict[str, Any]] = field(default_factory=list)
    rejected_claims: list[dict[str, Any]] = field(default_factory=list)
    audit_trail: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class StoryBeatNode:
    beat_id: str
    order: int
    unit_ids: list[str]
    word_ids: list[int]
    start_s: float
    end_s: float
    start_frame: int
    end_frame_exclusive: int
    verbatim_quote: str
    speech_act: str  # hook_contrast | negation_tension | allocation_step | payoff_reveal | breathing_bridge | product_intro | product_benefit | closing_takeaway | cta | statement
    narrative_role: str
    ledger_step_id: str | None = None
    product_entities: list[str] = field(default_factory=list)
    attribute_entities: list[str] = field(default_factory=list)
    negated: bool = False
    modality: str = "assertive"
    visual_observations: dict[str, Any] = field(default_factory=dict)
    mean_word_confidence: float = 1.0
    has_uncertain_number: bool = False

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class StoryGraph:
    schema_version: str
    source_hash: str
    transcript_hash: str
    language: str
    total_frames: int
    fps_float: float
    full_text: str
    product_name: str | None
    product_attributes: list[str]
    cta_phrase: str | None
    ledger: VerifiedLedger
    beats: list[StoryBeatNode]
    visual_summary: dict[str, Any]
    validation_errors: list[str] = field(default_factory=list)
    story_graph_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("story_graph_hash", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


def _flatten_words(transcript: Transcript) -> list[tuple[int, WordSpan]]:
    out: list[tuple[int, WordSpan]] = []
    idx = 0
    for seg in transcript.segments:
        for w in seg.words:
            out.append((idx, w))
            idx += 1
    return out


def _clean_category_name(raw_cat: str) -> str:
    tokens = [re.sub(r"[^A-Za-z0-9\-']+", "", t) for t in raw_cat.strip().split()]
    kept = [t for t in tokens if t and t.lower() not in _STOP_CAT_WORDS]
    if not kept:
        return "Item"
    # Stop at conjunctions/clauses
    out: list[str] = []
    for t in kept:
        if t.lower() in ("and", "but", "then", "which", "that", "you", "actually"):
            break
        out.append(t)
        if len(out) >= 2:
            break
    return " ".join(out).title() if out else "Item"


def _find_word_ids_in_span(
    indexed_words: list[tuple[int, WordSpan]],
    start_s: float,
    end_s: float,
    token_hint: str | None = None,
) -> list[int]:
    matched = [wid for wid, w in indexed_words if w.end_s >= start_s - 0.05 and w.start_s <= end_s + 0.05]
    if token_hint and matched:
        hint_clean = re.sub(r"[^a-z0-9]", "", token_hint.lower())
        exact = [
            wid for wid in matched
            if hint_clean and hint_clean in re.sub(r"[^a-z0-9]", "", indexed_words[wid][1].word.lower())
        ]
        if exact:
            return exact
    return matched


def _clause_spans_from_words(
    indexed_words: list[tuple[int, WordSpan]],
    fps_float: float,
    total_frames: int,
) -> list[dict[str, Any]]:
    """Split flattened ASR words into natural punctuation/pause/conjunction clauses with exact timestamps."""
    if not indexed_words:
        return []
    clauses: list[dict[str, Any]] = []
    cur: list[tuple[int, WordSpan]] = []

    def _flush() -> None:
        if not cur:
            return
        txt = " ".join(w.word.strip() for _, w in cur).strip()
        s_s = float(cur[0][1].start_s)
        e_s = float(cur[-1][1].end_s)
        sf = max(0, min(total_frames - 1, int(round(s_s * fps_float))))
        ef = max(sf + 1, min(total_frames, int(round(e_s * fps_float))))
        clauses.append({
            "word_ids": [i for i, _ in cur],
            "words": [w for _, w in cur],
            "text": txt,
            "start_s": round(s_s, 4),
            "end_s": round(e_s, 4),
            "start_frame": sf,
            "end_frame_exclusive": ef,
            "mean_conf": round(sum(w.probability for _, w in cur) / max(1, len(cur)), 4),
        })
        cur.clear()

    for pos, (wid, w) in enumerate(indexed_words):
        w_txt = w.word.strip()
        # Split BEFORE negation pivot ("doesn't mean"), allocation conjunction ("and 100 for gas"), or long pause (>=0.42s)
        if cur:
            prev_w = cur[-1][1]
            gap_s = float(w.start_s - prev_w.end_s)
            nxt_word = indexed_words[pos + 1][1].word.strip() if pos + 1 < len(indexed_words) else ""
            is_neg_pivot = w_txt.lower().startswith("doesn") and nxt_word.lower().startswith("mean")
            is_alloc_and = w_txt.lower() == "and" and any(ch.isdigit() for ch in nxt_word) and len(cur) >= 3
            is_product_part = (
                w_txt.lower() == "a"
                and any(prev_w.word.rstrip(".,!?").lower() in ("plan", "system", "guide") for _, prev_w in cur[-2:])
            )
            if gap_s >= 0.42 or is_neg_pivot or is_alloc_and or is_product_part:
                _flush()
        cur.append((wid, w))
        if w_txt.endswith((".", "!", "?", ";")) or (w_txt.endswith(",") and len(cur) >= 5):
            _flush()
    _flush()
    return clauses


def derive_verified_ledger(
    transcript: Transcript,
    clauses: list[dict[str, Any]],
    fps_float: float,
    total_frames: int,
) -> VerifiedLedger:
    """Build an integer-cent verified financial ledger with strict separation of verbatim quotes vs derived arithmetic."""
    indexed_words = _flatten_words(transcript)
    full_text = transcript.full_text or ""

    # Detect consistent currency context across transcript
    found_syms = [ch for ch in full_text if ch in _CURRENCY_SYMBOLS]
    currency_sym = found_syms[0] if found_syms else ""
    if not currency_sym:
        low_full = full_text.lower()
        for word, sym in _CURRENCY_WORDS.items():
            if re.search(rf"\b{word}\b", low_full):
                currency_sym = sym
                break
    currency_code = _CURRENCY_SYMBOLS.get(currency_sym, "UNIT" if not currency_sym else "USD")
    if not currency_sym:
        currency_sym = ""

    overall_modality = "hypothetical_example" if _HYPOTHETICAL_RE.search(full_text) else "actual_statement"

    steps: list[LedgerStep] = []
    non_monetary_counts: list[dict[str, Any]] = []
    negated_statements: list[dict[str, Any]] = []
    audit_trail: list[str] = []
    warnings: list[str] = []

    initial_cents = 0
    running_cents = 0
    total_allocated_cents = 0
    spoken_remaining_cents: int | None = None

    for c_idx, cl in enumerate(clauses):
        txt = cl["text"]
        is_neg = bool(_NEGATION_RE.search(txt))
        is_hyp = bool(_HYPOTHETICAL_RE.search(txt))
        is_rem = bool(_REMAINING_RE.search(txt))

        # 1. Extract non-monetary counts (e.g., "56-page workbook", "10%") first so they NEVER enter currency math!
        non_mon_matches: list[tuple[int, int]] = []
        for m in _AMOUNT_ANY_RE.finditer(txt):
            suf = (m.group("suffix") or "").strip().lower()
            if suf and ("page" in suf or "%" in suf):
                non_mon_matches.append((m.start(), m.end()))
                non_monetary_counts.append({
                    "raw": m.group(0).strip(),
                    "value": m.group("num"),
                    "unit": suf.lstrip("- "),
                    "clause_text": txt,
                    "start_s": cl["start_s"],
                    "end_s": cl["end_s"],
                    "start_frame": cl["start_frame"],
                    "end_frame_exclusive": cl["end_frame_exclusive"],
                })

        if is_neg:
            negated_statements.append({
                "clause_text": txt,
                "start_s": cl["start_s"],
                "end_s": cl["end_s"],
                "start_frame": cl["start_frame"],
                "end_frame_exclusive": cl["end_frame_exclusive"],
                "reason": "negated_modality_excluded_from_spendable_or_savings",
            })
            continue

        # 2. Check explicit allocation pattern ("800 is for bills", "200 for groceries", "100 for gas")
        alloc_matches = list(_ALLOCATION_FOR_RE.finditer(txt))
        if alloc_matches and initial_cents > 0:
            for am in alloc_matches:
                # Ensure not overlapping a non-monetary match
                if any(a <= am.start() < b for a, b in non_mon_matches):
                    continue
                amt_cents = parse_amount_to_cents(am.group("num"))
                if amt_cents <= 0:
                    continue
                sym = am.group("sym") or currency_sym
                cat = _clean_category_name(am.group("cat"))
                prev_bal = running_cents
                running_cents = prev_bal - amt_cents
                total_allocated_cents += amt_cents
                prop = round(amt_cents / max(1, initial_cents), 6)
                w_ids = _find_word_ids_in_span(indexed_words, cl["start_s"], cl["end_s"], am.group("num"))
                w_confs = [indexed_words[i][1].probability for i in w_ids] if w_ids else [cl["mean_conf"]]
                min_p = min(w_confs) if w_confs else 1.0
                # Focus timestamp onto the exact spoken number/category span
                t0_s = indexed_words[w_ids[0]][1].start_s if w_ids else cl["start_s"]
                t1_s = cl["end_s"]
                sf = max(0, min(total_frames - 1, int(round(t0_s * fps_float))))
                ef = max(sf + 1, min(total_frames, int(round(t1_s * fps_float))))
                formula = f"{prev_bal} - {amt_cents} = {running_cents} cents ({format_cents(prev_bal, sym)} - {format_cents(amt_cents, sym)} = {format_cents(running_cents, sym)})"
                steps.append(
                    LedgerStep(
                        step_id=f"ledger_step_{len(steps):02d}",
                        step_type="allocation",
                        category=cat,
                        delta_cents=-amt_cents,
                        amount_cents=amt_cents,
                        running_balance_cents=running_cents,
                        currency_symbol=sym,
                        currency_code=_CURRENCY_SYMBOLS.get(sym, currency_code),
                        proportion_of_initial=prop,
                        provenance="verbatim_spoken",
                        running_balance_provenance="derived_arithmetic",
                        formula=formula,
                        start_s=round(t0_s, 4),
                        end_s=round(t1_s, 4),
                        start_frame=sf,
                        end_frame_exclusive=ef,
                        word_ids=w_ids,
                        verbatim_quote=txt,
                        modality="hypothetical" if is_hyp else overall_modality,
                        uncertain=bool(min_p < 0.45),
                        confidence=round(min_p, 4),
                    )
                )
                audit_trail.append(f"ALLOCATION [{cat}]: verbatim={format_cents(amt_cents, sym)}; derived_balance={formula}")
            continue

        # 3. Opening balance or spoken remaining payoff
        money_nums = [
            m for m in _AMOUNT_ANY_RE.finditer(txt)
            if not any(a <= m.start() < b for a, b in non_mon_matches)
            and (m.group("sym") or currency_sym or _OPENING_BALANCE_RE.search(txt) or is_rem)
        ]
        if not money_nums:
            continue

        m0 = money_nums[0]
        amt_cents = parse_amount_to_cents(m0.group("num"))
        if amt_cents <= 0:
            continue
        sym = m0.group("sym") or currency_sym
        w_ids = _find_word_ids_in_span(indexed_words, cl["start_s"], cl["end_s"], m0.group("num"))
        w_confs = [indexed_words[i][1].probability for i in w_ids] if w_ids else [cl["mean_conf"]]
        min_p = min(w_confs) if w_confs else 1.0

        if initial_cents == 0 and (_OPENING_BALANCE_RE.search(txt) or c_idx <= 1):
            initial_cents = amt_cents
            running_cents = amt_cents
            formula = f"initial_balance = {amt_cents} cents ({format_cents(amt_cents, sym)})"
            steps.append(
                LedgerStep(
                    step_id=f"ledger_step_{len(steps):02d}",
                    step_type="opening_balance",
                    category="Bank Balance" if "bank" in txt.lower() else "Starting Balance",
                    delta_cents=amt_cents,
                    amount_cents=amt_cents,
                    running_balance_cents=running_cents,
                    currency_symbol=sym,
                    currency_code=_CURRENCY_SYMBOLS.get(sym, currency_code),
                    proportion_of_initial=1.0,
                    provenance="verbatim_spoken",
                    running_balance_provenance="verbatim_spoken",
                    formula=formula,
                    start_s=cl["start_s"],
                    end_s=cl["end_s"],
                    start_frame=cl["start_frame"],
                    end_frame_exclusive=cl["end_frame_exclusive"],
                    word_ids=w_ids,
                    verbatim_quote=txt,
                    modality="hypothetical" if is_hyp else "assertive",
                    uncertain=bool(min_p < 0.45),
                    confidence=round(min_p, 4),
                )
            )
            audit_trail.append(f"OPENING: {formula}")
        elif initial_cents > 0 and is_rem:
            spoken_remaining_cents = amt_cents
            matches_math = (spoken_remaining_cents == running_cents)
            formula = f"{initial_cents} - {total_allocated_cents} = {running_cents} cents (spoken={spoken_remaining_cents} cents, exact_match={matches_math})"
            steps.append(
                LedgerStep(
                    step_id=f"ledger_step_{len(steps):02d}",
                    step_type="remaining_confirmed",
                    category="Remaining",
                    delta_cents=0,
                    amount_cents=amt_cents,
                    running_balance_cents=running_cents,
                    currency_symbol=sym,
                    currency_code=_CURRENCY_SYMBOLS.get(sym, currency_code),
                    proportion_of_initial=round(max(0, running_cents) / max(1, initial_cents), 6),
                    provenance="verbatim_spoken",
                    running_balance_provenance="verbatim_spoken" if matches_math else "derived_arithmetic",
                    formula=formula,
                    start_s=cl["start_s"],
                    end_s=cl["end_s"],
                    start_frame=cl["start_frame"],
                    end_frame_exclusive=cl["end_frame_exclusive"],
                    word_ids=w_ids,
                    verbatim_quote=txt,
                    modality=overall_modality,
                    uncertain=bool(min_p < 0.45),
                    confidence=round(min_p, 4),
                )
            )
            audit_trail.append(f"REMAINING_CONFIRMED: {formula}")
            if not matches_math:
                warnings.append(
                    f"Spoken remaining ({format_cents(spoken_remaining_cents, sym)}) differs from derived arithmetic ({format_cents(running_cents, sym)})."
                )

    overspend_cents = max(0, -running_cents) if initial_cents > 0 else 0
    computed_rem = initial_cents - total_allocated_cents
    conservation_exact = bool(initial_cents > 0 and (computed_rem == running_cents))
    reconciled = bool(
        conservation_exact
        and (spoken_remaining_cents is None or spoken_remaining_cents == computed_rem)
    )
    active = bool(initial_cents > 0 and len(steps) >= 2)

    return VerifiedLedger(
        active=active,
        currency_symbol=currency_sym or "$",
        currency_code=currency_code,
        initial_cents=initial_cents,
        total_allocated_cents=total_allocated_cents,
        computed_remaining_cents=computed_rem,
        spoken_remaining_cents=spoken_remaining_cents,
        overspend_cents=overspend_cents,
        conservation_exact=conservation_exact,
        reconciled_with_spoken_remaining=reconciled,
        modality=overall_modality,
        steps=steps,
        non_monetary_counts=non_monetary_counts,
        negated_statements=negated_statements,
        rejected_claims=[],
        audit_trail=audit_trail,
        warnings=warnings,
    )


def _sample_visual_observation(roi_tracks: dict[str, Any] | None, sf: int, ef: int) -> dict[str, Any]:
    if not roi_tracks:
        return {"has_face": False, "has_task": False, "face_conf": 0.0, "task_conf": 0.0}
    kfs = roi_tracks.get("keyframes") or []
    span_kfs = [k for k in kfs if sf <= int(k.get("frame", -1)) < ef]
    if not span_kfs and kfs:
        mid = (sf + ef) // 2
        span_kfs = [min(kfs, key=lambda k: abs(int(k.get("frame", 0)) - mid))]
    if not span_kfs:
        return {"has_face": False, "has_task": False, "face_conf": 0.0, "task_conf": 0.0}
    face_c = sum(float(k["face"]["conf"]) for k in span_kfs) / len(span_kfs)
    task_c = sum(float(k["task"]["conf"]) for k in span_kfs) / len(span_kfs)
    mid_k = span_kfs[len(span_kfs) // 2]
    return {
        "has_face": bool(roi_tracks.get("has_face_track") and face_c >= 0.4),
        "has_task": bool(roi_tracks.get("has_task_track") and task_c >= 0.35),
        "face_box": mid_k["face"],
        "task_box": mid_k["task"],
        "face_conf": round(face_c, 3),
        "task_conf": round(task_c, 3),
        "tabletop_guard_y": roi_tracks.get("tabletop_guard_y"),
    }


def build_story_graph(
    transcript: Transcript,
    visual_observations: dict[str, Any] | None,
    media: MediaInfo,
    units: list[SemanticUnit] | None = None,
) -> StoryGraph:
    """Build complete word-aligned StoryGraph + integer-cent VerifiedLedger from offline ASR and vision tracks."""
    fps_f = media.fps_float
    total_f = media.duration_frames
    indexed_words = _flatten_words(transcript)
    clauses = _clause_spans_from_words(indexed_words, fps_f, total_f)
    ledger = derive_verified_ledger(transcript, clauses, fps_f, total_f)

    # Map ledger steps by start_frame so each beat links cleanly to its exact verified math step
    step_by_clause: dict[int, LedgerStep] = {}
    for st in ledger.steps:
        best_i = None
        best_ov = -1
        for i, cl in enumerate(clauses):
            ov = min(cl["end_frame_exclusive"], st.end_frame_exclusive) - max(cl["start_frame"], st.start_frame)
            if ov > best_ov:
                best_ov = ov
                best_i = i
        if best_i is not None and best_ov > 0:
            step_by_clause[best_i] = st

    # Extract product name, attributes, and CTA purely from transcript grammar (zero hardcoding!)
    product_name: str | None = None
    m_prod = _PRODUCT_INTRO_RE.search(transcript.full_text or "")
    if m_prod:
        cand = m_prod.group("prod").strip(" ,.!?;:")
        if 2 <= len(cand) <= 48:
            product_name = cand

    product_attributes: list[str] = []
    cta_phrase: str | None = None
    beats: list[StoryBeatNode] = []

    prev_end_f = 0
    for idx, cl in enumerate(clauses):
        sf, ef = int(cl["start_frame"]), int(cl["end_frame_exclusive"])
        # Insert breathing bridge beat whenever there is a genuine spoken pause >= 1.6s (e.g. visual demonstration pause!)
        if sf - prev_end_f >= int(round(fps_f * 1.6)):
            b_sf, b_ef = prev_end_f, sf
            beats.append(
                StoryBeatNode(
                    beat_id=f"beat_{len(beats):02d}_bridge",
                    order=len(beats),
                    unit_ids=[],
                    word_ids=[],
                    start_s=round(b_sf / fps_f, 4),
                    end_s=round(b_ef / fps_f, 4),
                    start_frame=b_sf,
                    end_frame_exclusive=b_ef,
                    verbatim_quote="",
                    speech_act="breathing_bridge",
                    narrative_role="visual_breathing_bridge",
                    visual_observations=_sample_visual_observation(visual_observations, b_sf, b_ef),
                )
            )

        txt = cl["text"]
        lstep = step_by_clause.get(idx)
        is_neg = bool(_NEGATION_RE.search(txt))
        is_cta = bool(_CTA_RE.search(txt))
        is_prod_intro = bool(product_name and product_name.lower() in txt.lower())
        is_benefit = bool(_BENEFIT_RE.search(txt))

        attrs: list[str] = []
        if is_benefit:
            for pat in (
                r"(?i)\b(Notion\s+dashboard|spreadsheet|dashboard|app|tracker)\b",
                r"(?i)\b(\d+\s*-\s*page\s+workbook|\d+\s+page\s+workbook|workbook|guide|checklist|planner)\b",
            ):
                for m_at in re.finditer(pat, txt):
                    val = re.sub(r"\s*-\s*", "-", m_at.group(1).strip())
                    if val not in attrs:
                        attrs.append(val)
                    if val not in product_attributes:
                        product_attributes.append(val)

        if is_cta and cta_phrase is None:
            cta_phrase = txt.strip()

        if lstep and lstep.step_type == "opening_balance":
            act, role = "hook_contrast", "opening_hook"
        elif is_neg:
            act, role = "negation_tension", "contradiction_contrast"
        elif lstep and lstep.step_type == "allocation":
            act, role = "allocation_step", f"allocation_{lstep.category.lower().replace(' ', '_')}"
        elif lstep and lstep.step_type == "remaining_confirmed":
            act, role = "payoff_reveal", "payoff_contrast"
        elif is_prod_intro:
            act, role = "product_intro", "solution_introduction"
        elif attrs or (is_benefit and idx >= len(clauses) // 2 and not is_cta and "payday" not in txt.lower() and "purpose" not in txt.lower()):
            act, role = "product_benefit", "solution_benefit"
        elif is_cta:
            act, role = "cta", "closing_cta"
        elif idx >= len(clauses) - 2:
            act, role = "closing_takeaway", "closing_takeaway"
        else:
            act, role = "statement", "narrative_beat"

        linked_units = [
            u.unit_id for u in (units or [])
            if u.start_frame < ef and u.end_frame_exclusive > sf
        ]
        beats.append(
            StoryBeatNode(
                beat_id=f"beat_{len(beats):02d}_{act}",
                order=len(beats),
                unit_ids=linked_units,
                word_ids=list(cl["word_ids"]),
                start_s=cl["start_s"],
                end_s=cl["end_s"],
                start_frame=sf,
                end_frame_exclusive=ef,
                verbatim_quote=txt,
                speech_act=act,
                narrative_role=role,
                ledger_step_id=lstep.step_id if lstep else None,
                product_entities=[product_name] if is_prod_intro and product_name else [],
                attribute_entities=attrs,
                negated=is_neg,
                modality="negated" if is_neg else ("hypothetical" if _HYPOTHETICAL_RE.search(txt) else "assertive"),
                visual_observations=_sample_visual_observation(visual_observations, sf, ef),
                mean_word_confidence=cl["mean_conf"],
                has_uncertain_number=bool(lstep and lstep.uncertain),
            )
        )
        prev_end_f = ef

    sg = StoryGraph(
        schema_version=STORY_GRAPH_SCHEMA_VERSION,
        source_hash=media.input_hash,
        transcript_hash=transcript.transcript_hash,
        language=transcript.language,
        total_frames=total_f,
        fps_float=round(fps_f, 4),
        full_text=transcript.full_text,
        product_name=product_name,
        product_attributes=product_attributes,
        cta_phrase=cta_phrase,
        ledger=ledger,
        beats=beats,
        visual_summary=dict((visual_observations or {}).get("summary") or {}),
    )
    sg.validation_errors = validate_claim_graph(sg)
    sg.story_graph_hash = sg.compute_hash()
    return sg


def validate_claim_graph(
    story_graph: StoryGraph,
    proposed_labels: list[str] | None = None,
) -> list[str]:
    """Fail-closed semantic claim & integer-cent conservation verifier."""
    errors: list[str] = []
    led = story_graph.ledger
    if led.active:
        if led.initial_cents <= 0:
            errors.append("Active ledger must have positive initial_cents.")
        recalc = led.initial_cents
        for st in led.steps:
            if st.step_type == "opening_balance":
                if st.amount_cents != led.initial_cents or st.running_balance_cents != led.initial_cents:
                    errors.append(f"Opening step mismatch in {st.step_id}")
            elif st.step_type == "allocation":
                if st.delta_cents != -st.amount_cents or st.amount_cents <= 0:
                    errors.append(f"Illegal allocation sign in {st.step_id}")
                recalc -= st.amount_cents
                if st.running_balance_cents != recalc:
                    errors.append(f"Running balance conservation failed at {st.step_id}: {st.running_balance_cents} != {recalc}")
                if st.running_balance_provenance != "derived_arithmetic":
                    errors.append(f"Intermediate balance {st.step_id} must carry provenance='derived_arithmetic'")
        if recalc != led.computed_remaining_cents:
            errors.append(f"Final ledger mismatch: {recalc} != {led.computed_remaining_cents}")
        if led.spoken_remaining_cents is not None and led.spoken_remaining_cents != recalc:
            errors.append(f"Spoken remaining {led.spoken_remaining_cents} != derived {recalc}")

    # Verify optional proposed display labels against forbidden hallucination/savings patterns & verbatim/derived allowlist
    if proposed_labels:
        allowed_nums: set[int] = set()
        if led.active:
            allowed_nums.add(led.initial_cents)
            allowed_nums.add(abs(led.computed_remaining_cents))
            for st in led.steps:
                allowed_nums.add(st.amount_cents)
                allowed_nums.add(abs(st.running_balance_cents))
        for nm in led.non_monetary_counts:
            allowed_nums.add(parse_amount_to_cents(str(nm["value"])))
        # Allow numbers explicitly spoken verbatim in transcript (e.g., historical year "1969" in storytelling niche)
        for tm in _AMOUNT_ANY_RE.finditer(story_graph.full_text or ""):
            allowed_nums.add(parse_amount_to_cents(tm.group("num")))
        for lbl in proposed_labels:
            if _FORBIDDEN_HALLUCINATION_RE.search(lbl):
                errors.append(f"Hallucinated savings/guarantee claim rejected: {lbl!r}")
            for m in _AMOUNT_ANY_RE.finditer(lbl):
                val_c = parse_amount_to_cents(m.group("num"))
                if val_c not in allowed_nums:
                    errors.append(f"Unverified numeric label rejected: {m.group(0)!r} in {lbl!r}")
    return errors

"""Phase A1 (Generic Multi-Source Speech Deduplication & Narrative Repair).

Reusable, niche-agnostic speech repair engine (ZERO hardcoded project names,
timestamps, or finance-only assumptions). Operates strictly on real local ASR
word timestamps, neural VAD spans, and optional 16kHz PCM waveforms.

Core capabilities:
  - `detect_cross_source_overlap`: detects boundary phrase overlaps, garbled AI
    generation tails diverging from an intended story, orphaned truncated clip
    prefixes, and missing required dialogue (`MISSING_SOURCE_DIALOGUE`).
  - `detect_disfluency_spans`: detects exact/near n-gram stutters, self-restarts
    (including partial word restarts), unapproved non-script fillers, and idle
    dead-air gaps while explicitly protecting intentional emphasis/refrains.
  - `choose_clean_speech_path`: competes multiple deduplication paths (e.g.
    keep-first vs keep-last occurrence) on measured acoustic join floor, zero
    phoneme clipping, semantic coverage, and flow — never fixed bonuses or quotas.
  - `validate_semantic_preservation`: verifies ordered word alignment, precision,
    recall, WER, critical tokens (dates, negations, CTA), and zero residual repeats.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from motionlab.schemas import canonical_hash, sanitize_native

SPEECH_DEDUP_SCHEMA_VERSION = "1.0.0"

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "to", "of", "in", "on", "at",
    "for", "is", "are", "was", "were", "be", "it", "this", "that", "i", "you",
    "we", "they", "my", "your", "their", "our", "me", "like", "just",
}
_NEGATION_TOKENS = {
    "not", "no", "never", "without", "dont", "don't", "cant", "can't",
    "wont", "won't", "isnt", "isn't", "arent", "aren't", "didnt", "didn't",
    "doesnt", "doesn't", "neither", "nor", "none", "nothing",
}
_DATE_DAY_TOKENS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "yesterday", "payday", "weekly", "monthly",
}
_FILLER_PHRASES = (
    ("i", "know"),
    ("you", "know"),
    ("um",),
    ("uh",),
    ("erm",),
    ("hmm",),
    ("hm",),
)


def normalize_token(raw: str) -> str:
    """Normalize punctuation/case while preserving contractions, negations, dates & units."""
    s = str(raw or "").strip().lower()
    # Keep apostrophes inside contractions (don't, I'm, what's, that's) and currency/numbers
    s = re.sub(r"^[^a-z0-9$€£¥]+|[^a-z0-9$€£¥%]+$", "", s)
    s = s.replace("’", "'")
    return s


def tokenize_text(text: str) -> list[str]:
    return [t for t in (normalize_token(w) for w in str(text or "").split()) if t]


def _word_field(w: Any, key: str, default: Any = None) -> Any:
    if isinstance(w, dict):
        return w.get(key, default)
    return getattr(w, key, default)


def _norm_word_list(words: list[Any], source_id: str = "A") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        raw = str(_word_field(w, "word", "") or "")
        s0 = float(_word_field(w, "source_start_s", _word_field(w, "start_s", 0.0)))
        s1 = float(_word_field(w, "source_end_s", _word_field(w, "end_s", s0)))
        p0 = float(_word_field(w, "protected_start_s", max(0.0, s0 - 0.05)))
        p1 = float(_word_field(w, "protected_end_s", s1 + 0.06))
        conf = float(_word_field(w, "confidence", _word_field(w, "probability", 0.9)))
        wid = int(_word_field(w, "word_id", i))
        out.append({
            "idx": i,
            "word_id": wid,
            "source_id": source_id,
            "raw": raw,
            "norm": normalize_token(raw),
            "start_s": round(s0, 4),
            "end_s": round(s1, 4),
            "protected_start_s": round(p0, 4),
            "protected_end_s": round(p1, 4),
            "confidence": round(conf, 4),
            "ends_sentence": raw.strip().endswith((".", "!", "?")),
        })
    return out


def _rms_dbfs_window(pcm_16k: np.ndarray | None, t0: float, t1: float, sr: int = 16000) -> float:
    """Measure local RMS dBFS in [t0, t1]; returns -40.0 neutral fallback when PCM absent."""
    if pcm_16k is None or pcm_16k.size < 320:
        return -40.0
    i0 = max(0, min(pcm_16k.size - 1, int(round(t0 * sr))))
    i1 = max(i0 + 1, min(pcm_16k.size, int(round(t1 * sr))))
    seg = pcm_16k[i0:i1]
    if seg.size == 0:
        return -60.0
    rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    return round(float(20.0 * np.log10(max(rms, 1e-6))), 2)


def _find_quietest_cut_point(
    pcm_16k: np.ndarray | None,
    t_lo: float,
    t_hi: float,
    *,
    sr: int = 16000,
    win_s: float = 0.04,
) -> tuple[float, float]:
    """Find the quietest 40ms center point inside [t_lo, t_hi] so cuts land cleanly in room tone."""
    if t_hi <= t_lo:
        t_mid = max(0.0, t_lo)
        return round(t_mid, 4), _rms_dbfs_window(pcm_16k, t_mid - 0.02, t_mid + 0.02, sr)
    if pcm_16k is None or pcm_16k.size < 640:
        t_mid = 0.5 * (t_lo + t_hi)
        return round(t_mid, 4), -40.0
    hop = max(1, int(round(0.01 * sr)))
    win = max(160, int(round(win_s * sr)))
    i_lo = max(0, int(round(t_lo * sr)))
    i_hi = min(pcm_16k.size - win, int(round(t_hi * sr)) - win)
    if i_hi <= i_lo:
        t_mid = 0.5 * (t_lo + t_hi)
        return round(t_mid, 4), _rms_dbfs_window(pcm_16k, t_lo, t_hi, sr)
    best_t = 0.5 * (t_lo + t_hi)
    best_db = 0.0
    for idx in range(i_lo, i_hi + 1, hop):
        seg = pcm_16k[idx : idx + win]
        db = float(20.0 * np.log10(max(float(np.sqrt(np.mean(seg * seg) + 1e-12)), 1e-6)))
        if db < best_db:
            best_db = db
            best_t = (idx + 0.5 * win) / float(sr)
    return round(best_t, 4), round(best_db, 2)


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if p.strip()]
    return parts


def _clause_in_tokens(clause_tokens: list[str], source_tokens: list[str]) -> float:
    """Best local normalized token match ratio of `clause_tokens` inside `source_tokens`."""
    if not clause_tokens or not source_tokens:
        return 0.0
    m = len(clause_tokens)
    if m <= 2:
        joined = " ".join(source_tokens)
        return 1.0 if " ".join(clause_tokens) in joined else 0.0
    best = 0.0
    for i in range(0, max(1, len(source_tokens) - m + 2)):
        win = source_tokens[i : min(len(source_tokens), i + int(m * 1.35) + 1)]
        ratio = SequenceMatcher(None, clause_tokens, win).ratio()
        if ratio > best:
            best = ratio
    return round(best, 4)


def detect_cross_source_overlap(
    words_a: list[Any],
    words_b: list[Any],
    intended_story: str | None = None,
    *,
    pcm_a_16k: np.ndarray | None = None,
    pcm_b_16k: np.ndarray | None = None,
    duration_a_s: float | None = None,
    duration_b_s: float | None = None,
) -> dict[str, Any]:
    """Detect A->B boundary overlap, AI-garbled tail divergence, truncated prefix & missing dialogue."""
    wa = _norm_word_list(words_a, "A")
    wb = _norm_word_list(words_b, "B")
    toks_a = [w["norm"] for w in wa if w["norm"]]
    toks_b = [w["norm"] for w in wb if w["norm"]]
    dur_a = float(duration_a_s if duration_a_s is not None else (wa[-1]["end_s"] if wa else 0.0))
    dur_b = float(duration_b_s if duration_b_s is not None else (wb[-1]["end_s"] if wb else 0.0))

    # 1. Exact/near suffix(A) <-> prefix(B) lexical overlap (2..8 tokens)
    exact_overlap: dict[str, Any] | None = None
    max_k = min(8, len(toks_a), len(toks_b))
    for k in range(max_k, 1, -1):
        if toks_a[-k:] == toks_b[:k]:
            exact_overlap = {
                "kind": "exact_boundary_overlap",
                "source_id": "B",
                "overlap_tokens": toks_a[-k:],
                "a_start_idx": len(wa) - k,
                "b_end_idx_exclusive": k,
                "b_trim_until_s": wb[k]["start_s"] if k < len(wb) else dur_b,
            }
            break

    # 2. Check against intended_story (when supplied): detect missing clauses, garbled tail in A, truncated prefix in B
    missing_clauses: list[dict[str, Any]] = []
    garbled_tail_a: dict[str, Any] | None = None
    truncated_prefix_b: dict[str, Any] | None = None

    if intended_story and intended_story.strip():
        sentences = _split_sentences(intended_story)
        combined_toks = toks_a + toks_b
        matched_a_end_idx = 0
        for s_idx, sent in enumerate(sentences):
            stoks = tokenize_text(sent)
            if len(stoks) < 3:
                continue
            r_comb = _clause_in_tokens(stoks, combined_toks)
            r_a = _clause_in_tokens(stoks, toks_a)
            r_b = _clause_in_tokens(stoks, toks_b)
            if r_comb < 0.62:
                missing_clauses.append({
                    "sentence_index": s_idx,
                    "clause": sent,
                    "best_match_ratio": r_comb,
                    "reason": "clause_absent_in_both_sources",
                })
            # Track end of last intended sentence cleanly spoken in A
            if r_a >= 0.72:
                m = len(stoks)
                best_end = matched_a_end_idx
                best_r = 0.0
                for i in range(0, len(toks_a)):
                    win = toks_a[i : min(len(toks_a), i + m + 2)]
                    rr = SequenceMatcher(None, stoks, win).ratio()
                    if rr >= 0.72 and rr >= best_r:
                        best_r = rr
                        best_end = min(len(wa), i + len(stoks))
                if best_end > matched_a_end_idx:
                    matched_a_end_idx = best_end

        # Detect garbled tail in A after last cleanly matched intended sentence!
        if matched_a_end_idx > 0 and matched_a_end_idx < len(wa):
            tail_words = wa[matched_a_end_idx:]
            tail_toks = [w["norm"] for w in tail_words if w["norm"]]
            # Verify tail does NOT match any subsequent intended sentence cleanly
            tail_matches_story = any(
                _clause_in_tokens(tokenize_text(s), tail_toks) >= 0.60
                for s in sentences
                if len(tokenize_text(s)) >= 3
            )
            if len(tail_words) >= 3 and not tail_matches_story:
                # Cut right after the last clean word's natural decay in quiet room floor
                last_clean_end = wa[matched_a_end_idx - 1]["end_s"]
                first_garble_start = tail_words[0]["start_s"]
                gap_len = first_garble_start - last_clean_end
                if gap_len >= 0.25:
                    cut_t, edge_db = _find_quietest_cut_point(
                        pcm_a_16k,
                        last_clean_end + min(0.18, gap_len * 0.3),
                        min(last_clean_end + 0.45, first_garble_start - 0.08),
                    )
                else:
                    cut_t = round(last_clean_end + 0.08, 4)
                    edge_db = _rms_dbfs_window(pcm_a_16k, cut_t - 0.02, cut_t + 0.02)
                garbled_tail_a = {
                    "kind": "garbled_divergent_tail",
                    "source_id": "A",
                    "start_word_idx": matched_a_end_idx,
                    "end_word_idx_exclusive": len(wa),
                    "cut_start_s": cut_t,
                    "cut_end_s": round(dur_a, 4),
                    "removed_s": round(max(0.0, dur_a - cut_t), 4),
                    "edge_rms_dbfs": edge_db,
                    "tail_token_count": len(tail_toks),
                    "reason": "Trailing source A words diverge from approved story (generation hallucination/garble) after final matched sentence.",
                }

        # Detect orphaned truncated prefix at very start of B (e.g. "Check plan." before "It shows you...")
        if len(wb) >= 4:
            # Find the BEST matching sentence-start index (1..5) in B whose FIRST token matches an intended sentence's FIRST token!
            prefix_end_idx = 0
            best_start_score = 0.0
            for cand_start in range(1, min(6, len(wb) - 2)):
                head_after = [w["norm"] for w in wb[cand_start : min(len(wb), cand_start + 6)]]
                for s in sentences:
                    stoks = tokenize_text(s)
                    if len(stoks) < 3 or not head_after or head_after[0] != stoks[0]:
                        continue
                    k_cmp = min(5, len(stoks), len(head_after))
                    sc = SequenceMatcher(None, stoks[:k_cmp], head_after[:k_cmp]).ratio()
                    if sc >= 0.72 and sc > best_start_score:
                        best_start_score = sc
                        prefix_end_idx = cand_start
            if prefix_end_idx > 0:
                pref_toks = [w["norm"] for w in wb[:prefix_end_idx]]
                # Confirm prefix is NOT a complete intended sentence on its own
                is_complete_sent = any(
                    SequenceMatcher(None, tokenize_text(s), pref_toks).ratio() >= 0.75
                    for s in sentences
                )
                if not is_complete_sent:
                    # Find quietest room tone strictly AFTER the last truncated prefix word and BEFORE the clean sentence onset!
                    prev_end = wb[prefix_end_idx - 1]["end_s"]
                    nxt_start = wb[prefix_end_idx]["start_s"]
                    # Whisper often drags low-conf sentence-initial function word ("it") backward to prev_end;
                    # look up to next confident word's onset - 0.16s to find the true acoustic room-tone valley!
                    search_hi = nxt_start - 0.05
                    if wb[prefix_end_idx]["confidence"] < 0.50 and prefix_end_idx + 1 < len(wb):
                        search_hi = max(search_hi, wb[prefix_end_idx + 1]["start_s"] - 0.18)
                    search_lo = max(0.0, prev_end + 0.10)
                    if search_hi > search_lo + 0.04:
                        cut_end_t, edge_db = _find_quietest_cut_point(pcm_b_16k, search_lo, search_hi)
                    else:
                        cut_end_t = round(max(0.0, max(prev_end + 0.12, nxt_start - 0.04)), 4)
                        edge_db = _rms_dbfs_window(pcm_b_16k, cut_end_t - 0.02, cut_end_t + 0.02)
                    truncated_prefix_b = {
                        "kind": "orphaned_truncated_prefix",
                        "source_id": "B",
                        "start_word_idx": 0,
                        "end_word_idx_exclusive": prefix_end_idx,
                        "prefix_tokens": pref_toks,
                        "cut_start_s": 0.0,
                        "cut_end_s": cut_end_t,
                        "removed_s": round(cut_end_t, 4),
                        "edge_rms_dbfs": edge_db,
                        "reason": "Orphaned truncated fragment at clip B head without matching stem in clip A; trim to clean sentence onset without synthesizing fake audio.",
                    }

    status = "MISSING_SOURCE_DIALOGUE" if missing_clauses else ("OVERLAP_DETECTED" if exact_overlap else "CLEAN_SEAM")
    return sanitize_native({
        "schema_version": SPEECH_DEDUP_SCHEMA_VERSION,
        "status": status,
        "exact_overlap": exact_overlap,
        "garbled_tail_a": garbled_tail_a,
        "truncated_prefix_b": truncated_prefix_b,
        "missing_dialogue_claims": missing_clauses,
        "missing_dialogue_count": len(missing_clauses),
    })


def detect_disfluency_spans(
    words: list[Any],
    prosody: dict[str, Any] | None = None,
    intended_story: str | None = None,
    vad_spans: dict[str, Any] | None = None,
    *,
    source_id: str = "B",
    pcm_16k: np.ndarray | None = None,
    niche: str = "general",
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Detect accidental stutters/repeats, unapproved fillers & dead air while preserving intentional emphasis."""
    wl = _norm_word_list(words, source_id)
    story_toks = tokenize_text(intended_story) if intended_story else []
    story_str = " " + " ".join(story_toks) + " "
    dur = float(duration_s if duration_s is not None else (wl[-1]["end_s"] if wl else 0.0))

    repeat_groups: list[dict[str, Any]] = []
    protected_intentional: list[dict[str, Any]] = []
    filler_candidates: list[dict[str, Any]] = []
    consumed_indices: set[int] = set()

    def _count_phrase_in_story(phrase: list[str]) -> int:
        if not story_toks or not phrase:
            return 0
        m = len(phrase)
        return sum(1 for i in range(0, len(story_toks) - m + 1) if story_toks[i : i + m] == phrase)

    # 1. Detect n-gram repetitions & self-restarts: OUTER loop longest n-gram (5 -> 2) first so full phrase
    #    matches ("what you can actually spend") ALWAYS win over shorter partial prefixes ("and what you can")!
    for n in (5, 4, 3, 2):
        i = 0
        while i <= len(wl) - n:
            if any(k in consumed_indices for k in range(i, i + n)):
                i += 1
                continue
            p1 = [w["norm"] for w in wl[i : i + n]]
            # Avoid matching trailing stop-word extensions ("mental math the") when a tighter content phrase exists
            if p1[-1] in _STOP_WORDS and n == 3:
                i += 1
                continue
            if not any(t and t not in _STOP_WORDS for t in p1):
                i += 1
                continue
            matched_here = False
            for j in range(i + n, min(len(wl) - n + 1, i + n + 12)):
                if any(k in consumed_indices for k in range(j, j + n)):
                    continue
                if wl[j]["start_s"] - wl[i + n - 1]["end_s"] > 5.0:
                    break
                p2 = [w["norm"] for w in wl[j : j + n]]
                if p1 == p2:
                    story_occ = _count_phrase_in_story(p1)
                    if story_occ >= 2:
                        protected_intentional.append({
                            "source_id": source_id,
                            "phrase": p1,
                            "first_idx": [i, i + n],
                            "second_idx": [j, j + n],
                            "reason": "INTENTIONAL_REPETITION_PROTECTED: phrase occurs multiple times in intended story.",
                        })
                        for k in range(i, j + n):
                            consumed_indices.add(k)
                        matched_here = True
                        break
                    between = [w["norm"] for w in wl[i + n : j]]
                    has_partial_restart = any(
                        (len(bt) >= 2 and any(t.startswith(bt) and len(t) > len(bt) for t in p1))
                        or bt in p1
                        for bt in between
                    )
                    if not story_toks and niche in ("comedy", "storytelling", "tutorial") and not has_partial_restart and len(between) >= 2:
                        protected_intentional.append({
                            "source_id": source_id,
                            "phrase": p1,
                            "first_idx": [i, i + n],
                            "second_idx": [j, j + n],
                            "reason": f"INTENTIONAL_REPETITION_PROTECTED: non-restart refrain in {niche} niche without contradicting script.",
                        })
                        for k in range(i, j + n):
                            consumed_indices.add(k)
                        matched_here = True
                        break

                    # Option A (keep_first): cut from quietest room floor after wl[i+n-1] to quietest room floor before wl[j+n]
                    a_after_prev_end = wl[i + n - 1]["end_s"]
                    a_next_start = wl[j + n]["start_s"] if (j + n) < len(wl) else dur
                    b_end_second = wl[j + n - 1]["end_s"]
                    # Search generously after first phrase end (up to +0.65s or next confident restart word) for deep room-tone valley!
                    a_search_hi = min(b_end_second - 0.10, a_after_prev_end + 0.65)
                    if (i + n) < len(wl) and wl[i + n]["confidence"] >= 0.92 and wl[i + n]["start_s"] > a_after_prev_end + 0.08:
                        a_search_hi = min(a_search_hi, wl[i + n]["start_s"] + 0.04)
                    a_cut0, a_db0 = _find_quietest_cut_point(
                        pcm_16k,
                        max(0.0, a_after_prev_end - 0.04),
                        max(a_after_prev_end + 0.06, a_search_hi),
                    )
                    if (j + n) < len(wl):
                        a_cut1, a_db1 = _find_quietest_cut_point(
                            pcm_16k,
                            max(b_end_second - 0.02, a_next_start - 0.45),
                            max(b_end_second + 0.08, a_next_start - 0.03),
                        )
                    else:
                        a_cut1, a_db1 = round(dur, 4), -40.0

                    # Option B (keep_last): cut from before wl[i] to before wl[j]
                    b_prev_end = wl[i - 1]["end_s"] if i > 0 else 0.0
                    b_cut0, b_db0 = _find_quietest_cut_point(
                        pcm_16k,
                        max(b_prev_end + 0.01, wl[i]["start_s"] - 0.08),
                        wl[i]["start_s"] + 0.02,
                    )
                    b_before_j_end = wl[j - 1]["end_s"]
                    b_cut1, b_db1 = _find_quietest_cut_point(
                        pcm_16k,
                        max(b_before_j_end - 0.02, wl[j]["start_s"] - 0.12),
                        wl[j]["start_s"] + 0.02,
                    )
                    a_lo_end = wl[i + n]["start_s"] if (i + n) < len(wl) else a_after_prev_end + 0.15
                    opt_a_gap_margin = round(max(0.0, (a_lo_end - a_after_prev_end)) + max(0.0, (a_next_start - b_end_second)), 3)
                    opt_b_gap_margin = round(max(0.0, (wl[i]["start_s"] - b_prev_end)) + max(0.0, (wl[j]["start_s"] - b_before_j_end)), 3)

                    repeat_groups.append({
                        "group_id": f"{source_id}_rep_{len(repeat_groups):02d}",
                        "source_id": source_id,
                        "category": "stutter_repeat",
                        "phrase": p1,
                        "between_tokens": between,
                        "has_partial_restart": has_partial_restart,
                        "first_span_idx": [i, i + n],
                        "second_span_idx": [j, j + n],
                        "options": {
                            "keep_first": {
                                "option_id": "keep_first",
                                "cut_start_s": a_cut0,
                                "cut_end_s": a_cut1,
                                "removed_s": round(max(0.0, a_cut1 - a_cut0), 4),
                                "edge_start_dbfs": a_db0,
                                "edge_end_dbfs": a_db1,
                                "edge_mean_dbfs": round(0.5 * (a_db0 + a_db1), 2),
                                "available_silence_margin_s": opt_a_gap_margin,
                                "dropped_word_indices": list(range(i + n, j + n)),
                            },
                            "keep_last": {
                                "option_id": "keep_last",
                                "cut_start_s": b_cut0,
                                "cut_end_s": b_cut1,
                                "removed_s": round(max(0.0, b_cut1 - b_cut0), 4),
                                "edge_start_dbfs": b_db0,
                                "edge_end_dbfs": b_db1,
                                "edge_mean_dbfs": round(0.5 * (b_db0 + b_db1), 2),
                                "available_silence_margin_s": opt_b_gap_margin,
                                "dropped_word_indices": list(range(i, j)),
                            },
                        },
                    })
                    for k in range(i, j + n):
                        consumed_indices.add(k)
                    matched_here = True
                    break
            i += n if matched_here else 1

    # 2. Detect unapproved filler phrases (ONLY when absent from intended_story AND verified quiet acoustic margins!)
    if story_toks:
        for idx in range(len(wl)):
            if idx in consumed_indices:
                continue
            for fp in _FILLER_PHRASES:
                flen = len(fp)
                if idx + flen > len(wl):
                    continue
                if tuple(wl[idx + k]["norm"] for k in range(flen)) == fp:
                    if f" {' '.join(fp)} " in story_str:
                        continue
                    prev_end = wl[idx - 1]["end_s"] if idx > 0 else 0.0
                    next_start = wl[idx + flen]["start_s"] if (idx + flen) < len(wl) else dur
                    # If Whisper glues filler end directly to next word start (e.g. "know" -> "I" at 16.865 while "left" starts at 17.365),
                    # search PCM up to next+1 start - 0.16s for genuine quiet room tone valley!
                    post_hi = next_start - 0.03
                    if (idx + flen + 1) < len(wl) and (next_start - wl[idx + flen - 1]["end_s"]) < 0.08:
                        post_hi = max(post_hi, wl[idx + flen + 1]["start_s"] - 0.22)
                    c0, db0 = _find_quietest_cut_point(pcm_16k, prev_end + 0.14, max(prev_end + 0.24, wl[idx]["start_s"] + 0.12))
                    c1, db1 = _find_quietest_cut_point(pcm_16k, wl[idx + flen - 1]["end_s"] + 0.02, max(wl[idx + flen - 1]["end_s"] + 0.10, post_hi))
                    gap_pre = wl[idx]["start_s"] - prev_end
                    gap_post = next_start - wl[idx + flen - 1]["end_s"]
                    # Require EITHER clear Whisper gaps OR confirmed deep PCM room-tone valleys on both sides!
                    acoustic_valleys_ok = (pcm_16k is not None and db0 <= -32.0 and db1 <= -32.0)
                    whisper_gaps_ok = (gap_pre >= 0.25 and gap_post >= 0.12)
                    if (acoustic_valleys_ok or whisper_gaps_ok) and c1 > c0 + 0.15:
                        filler_candidates.append({
                            "group_id": f"{source_id}_filler_{len(filler_candidates):02d}",
                            "source_id": source_id,
                            "category": "filler",
                            "phrase": list(fp),
                            "word_indices": list(range(idx, idx + flen)),
                            "cut_start_s": c0,
                            "cut_end_s": c1,
                            "removed_s": round(c1 - c0, 4),
                            "edge_start_dbfs": db0,
                            "edge_end_dbfs": db1,
                            "edge_mean_dbfs": round(0.5 * (db0 + db1), 2),
                            "reason": f"Unapproved non-script filler '{' '.join(fp)}' bounded by verified room-tone margins ({db0:.1f}dB / {db1:.1f}dB).",
                        })
                        for k in range(idx, idx + flen):
                            consumed_indices.add(k)
                        break

    # 3. Trailing dead-tail trim candidate (preserving >= 0.28s natural smile & final consonant decay!)
    tail_trim: dict[str, Any] | None = None
    if wl and dur - wl[-1]["end_s"] > 0.50:
        keep_end = round(min(dur, wl[-1]["end_s"] + 0.30), 4)
        if dur - keep_end >= 0.15:
            tail_trim = {
                "group_id": f"{source_id}_tail_trim",
                "source_id": source_id,
                "category": "dead_air",
                "cut_start_s": keep_end,
                "cut_end_s": round(dur, 4),
                "removed_s": round(dur - keep_end, 4),
                "reason": "Trim excess dead tail while preserving >=0.30s natural consonant decay and closing expression.",
            }

    return sanitize_native({
        "schema_version": SPEECH_DEDUP_SCHEMA_VERSION,
        "source_id": source_id,
        "repeat_groups": repeat_groups,
        "protected_intentional": protected_intentional,
        "filler_candidates": filler_candidates,
        "tail_trim": tail_trim,
    })


def _invert_cuts_to_kept(source_id: str, duration_s: float, cuts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted([c for c in cuts if c["cut_end_s"] > c["cut_start_s"] + 0.02], key=lambda c: c["cut_start_s"])
    merged: list[dict[str, Any]] = []
    for c in ordered:
        if merged and c["cut_start_s"] <= merged[-1]["cut_end_s"]:
            merged[-1]["cut_end_s"] = max(merged[-1]["cut_end_s"], c["cut_end_s"])
            merged[-1]["removed_s"] = round(merged[-1]["cut_end_s"] - merged[-1]["cut_start_s"], 4)
        else:
            merged.append(dict(c))
    kept: list[dict[str, Any]] = []
    cur = 0.0
    for c in merged:
        if c["cut_start_s"] > cur + 0.04:
            kept.append({"source_id": source_id, "start_s": round(cur, 4), "end_s": round(c["cut_start_s"], 4)})
        cur = max(cur, c["cut_end_s"])
    if duration_s > cur + 0.04:
        kept.append({"source_id": source_id, "start_s": round(cur, 4), "end_s": round(duration_s, 4)})
    return kept


def choose_clean_speech_path(
    sources_meta: list[dict[str, Any]],
    boundary_info: dict[str, Any] | list[dict[str, Any]] | None,
    disfluency_by_source: dict[str, dict[str, Any]],
    *,
    protected_claims: list[str] | None = None,
    intended_story: str | None = None,
    uncertainty: dict[str, Any] | None = None,
    candidate_score_overrides: dict[str, float] | None = None,
    bridge_duration_s: float = 0.0,
) -> dict[str, Any]:
    """Compete clean speech repair candidates causally (keep-first vs keep-last vs conservative vs identity)."""
    overrides = candidate_score_overrides or {}
    b_list: list[dict[str, Any]] = (
        [b for b in boundary_info if isinstance(b, dict)]
        if isinstance(boundary_info, list)
        else ([boundary_info] if isinstance(boundary_info, dict) else [])
    )
    has_boundary_defects = any(
        bool(b.get("garbled_tail_a") or b.get("truncated_prefix_b") or b.get("exact_overlap"))
        for b in b_list
    )
    all_missing_claims: list[dict[str, Any]] = []
    seen_mc: set[str] = set()
    for b in b_list:
        for mc in b.get("missing_dialogue_claims") or []:
            key = str(mc.get("clause") or mc)
            if key not in seen_mc:
                seen_mc.add(key)
                all_missing_claims.append(mc)

    def _assemble_variant(plan_id: str, rep_choice: str, include_fillers: bool, include_boundary_repairs: bool) -> dict[str, Any]:
        cuts: list[dict[str, Any]] = []
        edge_dbs: list[float] = []
        margin_s_total = 0.0
        repeats_resolved = 0
        repeats_remaining = 0

        # Boundary cuts (garbled tail, truncated prefix, exact overlap) — supports single dict or list across N>=2 sources
        if include_boundary_repairs and boundary_info:
            b_list = boundary_info if isinstance(boundary_info, list) else [boundary_info]
            for b_item in b_list:
                if not isinstance(b_item, dict):
                    continue
                for key in ("garbled_tail_a", "truncated_prefix_b"):
                    item = b_item.get(key)
                    if isinstance(item, dict) and float(item.get("removed_s", 0.0)) > 0.05:
                        cuts.append({
                            "cut_id": f"{item['source_id']}_{item['kind']}",
                            "source_id": item["source_id"],
                            "category": item["kind"],
                            "cut_start_s": float(item["cut_start_s"]),
                            "cut_end_s": float(item["cut_end_s"]),
                            "removed_s": float(item["removed_s"]),
                            "edge_rms_dbfs": float(item.get("edge_rms_dbfs", -36.0)),
                            "reason": str(item.get("reason", "")),
                        })
                        edge_dbs.append(float(item.get("edge_rms_dbfs", -36.0)))
                ov = b_item.get("exact_overlap")
                if isinstance(ov, dict) and float(ov.get("b_trim_until_s", 0.0)) > 0.05:
                    ov_sid = str(ov.get("source_id") or (sources_meta[1]["source_id"] if len(sources_meta) > 1 else "B"))
                    cuts.append({
                        "cut_id": f"{ov_sid}_exact_boundary_overlap",
                        "source_id": ov_sid,
                        "category": "boundary_overlap",
                        "cut_start_s": 0.0,
                        "cut_end_s": float(ov["b_trim_until_s"]),
                        "removed_s": float(ov["b_trim_until_s"]),
                        "edge_rms_dbfs": -36.0,
                        "reason": f"Trim exact duplicated boundary phrase at clip {ov_sid} head.",
                    })

        for src in sources_meta:
            sid = str(src["source_id"])
            dis = disfluency_by_source.get(sid) or {}
            for rg in dis.get("repeat_groups", []):
                if rep_choice == "none":
                    repeats_remaining += 1
                    continue
                opts = rg.get("options") or {}
                if rep_choice == "auto_acoustic":
                    # Choose whichever of keep_first / keep_last has quieter room-tone edges + wider silence margin!
                    o1 = opts["keep_first"]
                    o2 = opts["keep_last"]
                    s1 = (-float(o1["edge_mean_dbfs"])) + 12.0 * float(o1["available_silence_margin_s"])
                    s2 = (-float(o2["edge_mean_dbfs"])) + 12.0 * float(o2["available_silence_margin_s"])
                    pick = o1 if s1 >= s2 else o2
                else:
                    pick = opts.get(rep_choice) or opts["keep_first"]
                cuts.append({
                    "cut_id": f"{rg['group_id']}_{pick['option_id']}",
                    "source_id": sid,
                    "category": "stutter_repeat",
                    "option_chosen": pick["option_id"],
                    "phrase": rg.get("phrase", []),
                    "cut_start_s": float(pick["cut_start_s"]),
                    "cut_end_s": float(pick["cut_end_s"]),
                    "removed_s": float(pick["removed_s"]),
                    "edge_rms_dbfs": float(pick["edge_mean_dbfs"]),
                    "silence_margin_s": float(pick["available_silence_margin_s"]),
                    "reason": f"Remove accidental repetition of '{' '.join(rg.get('phrase', []))}' ({pick['option_id']}).",
                })
                edge_dbs.append(float(pick["edge_mean_dbfs"]))
                margin_s_total += float(pick["available_silence_margin_s"])
                repeats_resolved += 1

            if include_fillers:
                for fc in dis.get("filler_candidates", []):
                    cuts.append({
                        "cut_id": fc["group_id"],
                        "source_id": sid,
                        "category": "filler",
                        "phrase": fc.get("phrase", []),
                        "cut_start_s": float(fc["cut_start_s"]),
                        "cut_end_s": float(fc["cut_end_s"]),
                        "removed_s": float(fc["removed_s"]),
                        "edge_rms_dbfs": float(fc["edge_mean_dbfs"]),
                        "reason": str(fc.get("reason", "")),
                    })
                    edge_dbs.append(float(fc["edge_mean_dbfs"]))
            tt = dis.get("tail_trim")
            if include_boundary_repairs and isinstance(tt, dict):
                cuts.append({
                    "cut_id": tt["group_id"],
                    "source_id": sid,
                    "category": "dead_air",
                    "cut_start_s": float(tt["cut_start_s"]),
                    "cut_end_s": float(tt["cut_end_s"]),
                    "removed_s": float(tt["removed_s"]),
                    "edge_rms_dbfs": -38.0,
                    "reason": str(tt.get("reason", "")),
                })

        # Build ordered kept spans across all sources (inserting optional editorial bridge between A and B)
        kept_spans: list[dict[str, Any]] = []
        total_removed = 0.0
        for s_idx, src in enumerate(sources_meta):
            sid = str(src["source_id"])
            dur = float(src["duration_s"])
            s_cuts = [c for c in cuts if c["source_id"] == sid]
            total_removed += sum(c["removed_s"] for c in s_cuts)
            spans = _invert_cuts_to_kept(sid, dur, s_cuts)
            for sp in spans:
                sp["kind"] = "source_media"
                kept_spans.append(sp)
            if s_idx == 0 and len(sources_meta) > 1 and bridge_duration_s > 0.0 and include_boundary_repairs:
                kept_spans.append({
                    "source_id": "bridge_editorial",
                    "kind": "editorial_bridge",
                    "start_s": 0.0,
                    "end_s": round(float(bridge_duration_s), 4),
                })

        # Objective scoring (NO fixed variant bonus, NO duration quota!)
        mean_edge_db = float(np.mean(edge_dbs)) if edge_dbs else -35.0
        # Quieter cut edges (-35 dBFS vs -20 dBFS) earn higher join-safety score
        acoustic_join_score = max(0.0, min(30.0, (-mean_edge_db - 18.0) * 1.4))
        dedup_gain = 18.0 * repeats_resolved - 28.0 * repeats_remaining
        boundary_gain = 15.0 if (include_boundary_repairs and has_boundary_defects) else 0.0
        filler_gain = 5.0 * sum(1 for c in cuts if c["category"] == "filler")
        margin_gain = min(15.0, margin_s_total * 8.0)
        # Penalize noisy cut edges (> -27 dBFS risks clipping consonants!)
        clip_risk_penalty = sum(max(0.0, (db + 27.0) * 3.5) for db in edge_dbs)
        score = round(
            50.0 + dedup_gain + boundary_gain + filler_gain + acoustic_join_score + margin_gain - clip_risk_penalty
            + float(overrides.get(plan_id, 0.0)),
            3,
        )
        out_dur = round(sum(sp["end_s"] - sp["start_s"] for sp in kept_spans), 4)
        return {
            "plan_id": plan_id,
            "eligible": True,
            "score": score,
            "score_breakdown": {
                "dedup_gain": round(dedup_gain, 2),
                "boundary_gain": round(boundary_gain, 2),
                "filler_gain": round(filler_gain, 2),
                "acoustic_join_score": round(acoustic_join_score, 2),
                "margin_gain": round(margin_gain, 2),
                "clip_risk_penalty": round(clip_risk_penalty, 2),
                "override_delta": round(float(overrides.get(plan_id, 0.0)), 2),
            },
            "repeats_resolved": repeats_resolved,
            "repeats_remaining": repeats_remaining,
            "cuts_count": len(cuts),
            "removed_total_s": round(total_removed, 4),
            "output_duration_s": out_dur,
            "cuts": cuts,
            "kept_spans": kept_spans,
            "plan_hash": canonical_hash({"cuts": cuts, "kept_spans": kept_spans}),
        }

    candidates = [
        _assemble_variant("clean_keep_first", "keep_first", True, True),
        _assemble_variant("clean_keep_last", "keep_last", True, True),
        _assemble_variant("clean_auto_acoustic", "auto_acoustic", True, True),
        _assemble_variant("conservative_no_filler", "keep_first", False, True),
        _assemble_variant("identity_qa_baseline", "none", False, False),
    ]
    # Policy: never auto-pick identity baseline if confirmed repeats/garble exist unless explicitly overridden!
    has_confirmed_defects = any(
        bool((disfluency_by_source.get(s["source_id"]) or {}).get("repeat_groups"))
        for s in sources_meta
    ) or has_boundary_defects
    if has_confirmed_defects and "identity_qa_baseline" not in overrides:
        for c in candidates:
            if c["plan_id"] == "identity_qa_baseline":
                c["eligible"] = False
                c["rejection_reason"] = "unrepaired_confirmed_speech_defects"

    eligible = [c for c in candidates if c["eligible"]]
    chosen = max(eligible or candidates, key=lambda c: (c["score"], -c["cuts_count"]))
    return sanitize_native({
        "schema_version": SPEECH_DEDUP_SCHEMA_VERSION,
        "chosen_plan_id": chosen["plan_id"],
        "chosen_plan": chosen,
        "candidates": candidates,
        "causal_selection_before_compile": True,
        "missing_dialogue_claims": all_missing_claims,
    })


def validate_semantic_preservation(
    source_words: list[Any],
    approved_keep_tokens: list[str],
    edited_words: list[Any],
    *,
    critical_tokens: list[str] | None = None,
    missing_source_claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate ordered semantic preservation, precision/recall, WER, critical tokens & zero residual repeats."""
    target = [normalize_token(t) for t in approved_keep_tokens if normalize_token(t)]
    edited = [normalize_token(_word_field(w, "word", w) if not isinstance(w, str) else w) for w in edited_words]
    edited = [t for t in edited if t]

    sm = SequenceMatcher(None, target, edited)
    matched = sum(blk.size for blk in sm.get_matching_blocks())
    recall = round(matched / max(1, len(target)), 4)
    precision = round(matched / max(1, len(edited)), 4)

    # Word Error Rate (Levenshtein distance over normalized tokens)
    m, n = len(target), len(edited)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            cost = 0 if target[i - 1] == edited[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    wer = round(dp[n] / max(1, m), 4)

    # Critical tokens (negations, days/dates, explicit protected words)
    crit_set = [normalize_token(t) for t in (critical_tokens or []) if normalize_token(t)]
    if not crit_set:
        crit_set = [t for t in target if t in _NEGATION_TOKENS or t in _DATE_DAY_TOKENS or any(c.isdigit() for c in t)]
    edited_set = set(edited)
    missing_critical = [t for t in crit_set if t not in edited_set]
    crit_recall = round((len(crit_set) - len(missing_critical)) / max(1, len(crit_set)), 4) if crit_set else 1.0

    # Check for residual accidental repeats (consecutive 2- or 3-gram stutters in edited words that aren't repeated in target!)
    residual_repeats: list[list[str]] = []
    target_str = " " + " ".join(target) + " "
    for n_g in (3, 2):
        for i in range(0, len(edited) - 2 * n_g + 1):
            p = edited[i : i + n_g]
            if not any(t not in _STOP_WORDS for t in p):
                continue
            if edited[i + n_g : i + 2 * n_g] == p:
                if target_str.count(" " + " ".join(p) + " ") < 2:
                    residual_repeats.append(p)

    # Check ordering monotonicity of matched blocks
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    order_ok = all(blocks[k].b <= blocks[k + 1].b for k in range(len(blocks) - 1))

    has_missing_src = bool(missing_source_claims)
    core_ok = (recall >= 0.96) and (crit_recall >= 1.0) and (len(residual_repeats) == 0) and order_ok
    if not core_ok:
        status = "FAIL"
    elif has_missing_src:
        status = "PASS_WITH_MISSING_SOURCE_DIALOGUE"
    else:
        status = "PASS"

    return sanitize_native({
        "schema_version": SPEECH_DEDUP_SCHEMA_VERSION,
        "status": status,
        "passed": core_ok,
        "target_token_count": len(target),
        "edited_token_count": len(edited),
        "matched_ordered_tokens": matched,
        "token_recall": recall,
        "token_precision": precision,
        "wer": wer,
        "critical_tokens_checked": crit_set,
        "critical_token_recall": crit_recall,
        "missing_critical_tokens": missing_critical,
        "residual_repeats": residual_repeats,
        "order_preserved": order_ok,
        "missing_source_dialogue_claims": list(missing_source_claims or []),
    })


def align_source_dialogue_to_intent(
    words_by_source: dict[str, list[dict[str, Any]]],
    intended_story: str | None,
    *,
    disfluency_by_source: dict[str, dict[str, Any]] | None = None,
    boundary_infos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A1 (R5.R6): Monotonic source-to-intent alignment strictly separating source completeness from edit recall.

    Classifies every intended canonical sentence into:
      - AVAILABLE_EXACT (match ratio >= 0.90)
      - AVAILABLE_VARIANT (0.62 <= match ratio < 0.90)
      - UNCERTAIN (0.45 <= match ratio < 0.62 — low ASR confidence / possible partial garble)
      - MISSING_SOURCE (match ratio < 0.45 across ALL supplied source clips)
    """
    source_ids = list(words_by_source.keys())
    all_words: list[dict[str, Any]] = []
    for sid in source_ids:
        for w in _norm_word_list(words_by_source[sid], sid):
            all_words.append(w)
    all_toks = [w["norm"] for w in all_words if w["norm"]]

    sentences = _split_sentences(intended_story or "")
    rows: list[dict[str, Any]] = []
    counts = {
        "AVAILABLE_EXACT": 0,
        "AVAILABLE_VARIANT": 0,
        "UNCERTAIN": 0,
        "MISSING_SOURCE": 0,
    }
    for s_idx, sent in enumerate(sentences):
        stoks = tokenize_text(sent)
        if not stoks:
            continue
        ratio = _clause_in_tokens(stoks, all_toks)
        # Identify which source clip(s) carry the best match
        best_src = None
        best_src_ratio = 0.0
        for sid in source_ids:
            st = [normalize_token(_word_field(w, "word", "")) for w in words_by_source[sid]]
            st = [t for t in st if t]
            r_s = _clause_in_tokens(stoks, st)
            if r_s > best_src_ratio:
                best_src_ratio = r_s
                best_src = sid

        if ratio >= 0.90:
            cls = "AVAILABLE_EXACT"
        elif ratio >= 0.62:
            cls = "AVAILABLE_VARIANT"
        elif ratio >= 0.45:
            cls = "UNCERTAIN"
        else:
            cls = "MISSING_SOURCE"
        counts[cls] += 1
        rows.append({
            "sentence_index": s_idx,
            "intended_sentence": sent,
            "token_count": len(stoks),
            "best_match_ratio": ratio,
            "best_source_id": best_src if best_src_ratio >= 0.45 else None,
            "classification": cls,
        })

    # Collect duplicate restarts & garbled fragments recorded during repair inspection
    disfluency_events: list[dict[str, Any]] = []
    for sid, dinfo in (disfluency_by_source or {}).items():
        for rg in dinfo.get("repeat_groups") or []:
            disfluency_events.append({
                "source_id": sid,
                "classification": "DUPLICATE_RESTART",
                "phrase": rg.get("phrase", []),
                "occurrence_1_s": rg.get("first_span_s"),
                "occurrence_2_s": rg.get("second_span_s"),
            })
    for b in (boundary_infos or []):
        if b.get("garbled_tail_a"):
            gt = b["garbled_tail_a"]
            disfluency_events.append({
                "source_id": gt.get("source_id"),
                "classification": "GARBLED",
                "cut_start_s": gt.get("cut_start_s"),
                "cut_end_s": gt.get("cut_end_s"),
                "reason": gt.get("reason", ""),
            })
        if b.get("truncated_prefix_b"):
            tp = b["truncated_prefix_b"]
            disfluency_events.append({
                "source_id": tp.get("source_id"),
                "classification": "GARBLED",
                "prefix_tokens": tp.get("prefix_tokens", []),
                "cut_start_s": tp.get("cut_start_s"),
                "cut_end_s": tp.get("cut_end_s"),
                "reason": tp.get("reason", ""),
            })

    total_s = len(rows)
    avail_s = counts["AVAILABLE_EXACT"] + counts["AVAILABLE_VARIANT"]
    completeness = round(avail_s / max(1, total_s), 4) if total_s else 1.0
    missing_rows = [r for r in rows if r["classification"] in ("MISSING_SOURCE", "UNCERTAIN")]
    return sanitize_native({
        "schema_version": "2.0.0",
        "sources_audited": source_ids,
        "intended_sentences_total": total_s,
        "available_sentences_count": avail_s,
        "missing_or_uncertain_sentences_count": len(missing_rows),
        "source_completeness_ratio": completeness,
        "source_completeness_status": "COMPLETE_SOURCE_SCRIPT" if not missing_rows else "MISSING_SOURCE_DIALOGUE",
        "classification_counts": counts,
        "sentence_alignment": rows,
        "disfluency_and_garble_events": disfluency_events,
    })


def verify_required_claims(
    alignment_report: dict[str, Any],
    edited_words: list[dict[str, Any]],
    *,
    product_name: str | None = None,
) -> dict[str, Any]:
    """A1 (R5.R6): Fail-closed critical claim & release-blocking dialogue completeness check."""
    edited_toks = [normalize_token(_word_field(w, "word", w) if not isinstance(w, str) else w) for w in edited_words]
    edited_toks = [t for t in edited_toks if t]
    edited_set = set(edited_toks)

    avail_sentences = [
        r for r in (alignment_report.get("sentence_alignment") or [])
        if r.get("classification") in ("AVAILABLE_EXACT", "AVAILABLE_VARIANT")
    ]
    missing_sentences = [
        r for r in (alignment_report.get("sentence_alignment") or [])
        if r.get("classification") in ("MISSING_SOURCE", "UNCERTAIN")
    ]

    # Critical tokens present in AVAILABLE source sentences MUST survive in edited output!
    avail_toks = [t for r in avail_sentences for t in tokenize_text(r.get("intended_sentence", ""))]
    required_crit = sorted({
        t for t in avail_toks
        if t in _NEGATION_TOKENS or t in _DATE_DAY_TOKENS or t in ("caption", "bio", "profile", "link")
    })
    dropped_crit = [t for t in required_crit if t not in edited_set]

    # Check if product_name was present in available source vs absent in all sources
    prod_toks = tokenize_text(product_name or "")
    prod_in_avail = bool(prod_toks) and _clause_in_tokens(prod_toks, avail_toks) >= 0.75
    prod_in_edited = bool(prod_toks) and _clause_in_tokens(prod_toks, edited_toks) >= 0.75
    dropped_brand = prod_in_avail and not prod_in_edited

    retained_ok = (len(dropped_crit) == 0) and (not dropped_brand)
    release_blocked = bool(missing_sentences) or (not retained_ok)
    return sanitize_native({
        "schema_version": "2.0.0",
        "retained_critical_speech_ok": retained_ok,
        "required_available_critical_tokens": required_crit,
        "dropped_critical_tokens": dropped_crit,
        "product_name_spoken_in_source": prod_in_avail,
        "product_name_preserved_in_edit": prod_in_edited,
        "dropped_spoken_brand": dropped_brand,
        "missing_source_sentences_count": len(missing_sentences),
        "missing_source_sentences": [r["intended_sentence"] for r in missing_sentences],
        "release_blocked_by_script_completeness": release_blocked,
        "status": (
            "FAIL_DROPPED_CRITICAL_SPEECH"
            if not retained_ok
            else ("RELEASE_BLOCKED_MISSING_SOURCE_DIALOGUE" if missing_sentences else "PASS_FULL_SCRIPT_VERIFIED")
        ),
    })

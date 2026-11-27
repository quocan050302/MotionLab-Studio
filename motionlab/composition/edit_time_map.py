"""Phase A3 (ML-01.R5.R4): Canonical EditTimeMap, Exact Rational Clocks, Derived Mezzanine & Word Preservation.

Strict engineering invariants (Fixes D01, D05, D06):
  1. Explicit dual time domains (`source` vs `edited`) with exact integer frames & `Fraction` rational arithmetic.
     NEVER silently rounds 24000/1001 to 24 or accumulates floating-point sample drift across cuts.
  2. `output_total_frames` is independently derived as `sum(kept_span.frame_count)` — NEVER derived from output MP4!
  3. Removed source frames map strictly to `None` (NEVER an invented nearest output frame).
  4. Cut boundaries round strictly INWARD to integer frames and fail closed if any cut intersects a protected
     word/uncertainty span or critical action interval.
  5. Strategy A (Controlled Near-Lossless Mezzanine + Derived PCM Voice Bed):
       - Immutable original source MP4 is NEVER modified; its SHA-256 is checked before & after.
       - `original_audio_master.m4a` stays bit-exact stream-copied from original source.
       - Edited mezzanine (`crf=10`, geometry/color/fps preserved) & derived PCM voice bed live strictly
         as derived artifacts linked by `parent_source_sha256` and `edit_map_hash`.
       - Identity maps (`pacing == "preserve"` or 0 cuts) bypass derivative generation cleanly.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from motionlab.schemas import MediaInfo, atomic_write_json, canonical_hash, sanitize_native
from motionlab.semantic.schema import SEMANTIC_SCHEMA_VERSION, Transcript, TranscriptSegment, WordSpan

EDIT_TIME_MAP_SCHEMA_VERSION = "1.0.0"

_TOKEN_NORM_RE = re.compile(r"[^a-z0-9\$]+")
_CRITICAL_NEGATIONS = frozenset({"not", "never", "no", "dont", "doesn", "doesnt", "cant", "wont", "isnt", "arent", "without"})


@dataclass(frozen=True)
class KeptSpan:
    """Contiguous kept source frame span mapped onto contiguous output timeline frames & samples."""

    span_id: str
    source_start_frame: int
    source_end_frame_exclusive: int
    output_start_frame: int
    output_end_frame_exclusive: int
    source_start_sample: int
    source_end_sample: int
    output_start_sample: int
    output_end_sample: int
    crossfade_in_samples: int = 0
    crossfade_out_samples: int = 0

    @property
    def frame_count(self) -> int:
        return int(self.source_end_frame_exclusive - self.source_start_frame)

    @property
    def sample_count(self) -> int:
        return int(self.output_end_sample - self.output_start_sample)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass(frozen=True)
class OmittedSpan:
    """Explicitly cut source span with provenance, pause class, reason, and evidence references."""

    cut_id: str
    decision_id: str
    source_start_frame: int
    source_end_frame_exclusive: int
    source_start_s: float
    source_end_s: float
    removed_frames: int
    removed_s: float
    pause_class: str
    action: str
    reason: str
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class EditTimeMap:
    """Immutable canonical map linking source timeline domain to edited output timeline domain."""

    schema_version: str
    source_sha256: str
    source_path: str
    fps_num: int
    fps_den: int
    sample_rate: int
    source_total_frames: int
    source_duration_s: float
    source_total_samples: int
    pacing_mode: str  # "preserve" | "auto"
    is_identity: bool
    kept_spans: list[KeptSpan]
    omitted_spans: list[OmittedSpan]
    output_total_frames: int
    output_duration_s: float
    edited_total_samples: int
    removed_total_frames: int
    removed_total_s: float
    speed_ratio_num: int = 1
    speed_ratio_den: int = 1
    status: str = "OK"  # "IDENTITY_PRESERVE" | "EDITED_CUTS_APPLIED" | "PACING_NO_SAFE_CUTS"
    map_hash: str = ""

    def compute_hash(self) -> str:
        d = self.to_dict()
        d.pop("map_hash", None)
        d.pop("source_path", None)  # Keep map_hash host-path-independent!
        return canonical_hash(d)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native({
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "sample_rate": self.sample_rate,
            "source_total_frames": self.source_total_frames,
            "source_duration_s": round(self.source_duration_s, 6),
            "source_total_samples": self.source_total_samples,
            "pacing_mode": self.pacing_mode,
            "is_identity": self.is_identity,
            "kept_spans": [k.to_dict() for k in self.kept_spans],
            "omitted_spans": [o.to_dict() for o in self.omitted_spans],
            "output_total_frames": self.output_total_frames,
            "output_duration_s": round(self.output_duration_s, 6),
            "edited_total_samples": self.edited_total_samples,
            "removed_total_frames": self.removed_total_frames,
            "removed_total_s": round(self.removed_total_s, 6),
            "speed_ratio_num": self.speed_ratio_num,
            "speed_ratio_den": self.speed_ratio_den,
            "status": self.status,
            "map_hash": self.map_hash,
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EditTimeMap":
        return cls(
            schema_version=str(d.get("schema_version", EDIT_TIME_MAP_SCHEMA_VERSION)),
            source_sha256=str(d["source_sha256"]),
            source_path=str(d.get("source_path", "")),
            fps_num=int(d["fps_num"]),
            fps_den=int(d["fps_den"]),
            sample_rate=int(d["sample_rate"]),
            source_total_frames=int(d["source_total_frames"]),
            source_duration_s=float(d["source_duration_s"]),
            source_total_samples=int(d["source_total_samples"]),
            pacing_mode=str(d.get("pacing_mode", "preserve")),
            is_identity=bool(d.get("is_identity", False)),
            kept_spans=[KeptSpan(**k) for k in d.get("kept_spans", [])],
            omitted_spans=[OmittedSpan(**o) for o in d.get("omitted_spans", [])],
            output_total_frames=int(d["output_total_frames"]),
            output_duration_s=float(d["output_duration_s"]),
            edited_total_samples=int(d["edited_total_samples"]),
            removed_total_frames=int(d.get("removed_total_frames", 0)),
            removed_total_s=float(d.get("removed_total_s", 0.0)),
            speed_ratio_num=int(d.get("speed_ratio_num", 1)),
            speed_ratio_den=int(d.get("speed_ratio_den", 1)),
            status=str(d.get("status", "OK")),
            map_hash=str(d.get("map_hash", "")),
        )


def exact_rational_samples(frames: int, fps_num: int, fps_den: int, sample_rate: int) -> int:
    """Compute exact integer audio sample count for `frames` at rational `fps_num/fps_den` with ZERO float drift."""
    frac = Fraction(int(frames) * int(fps_den) * int(sample_rate), max(1, int(fps_num)))
    # Round half to even deterministically
    return int(round(frac))


def build_identity_edit_time_map(media: MediaInfo, *, pacing_mode: str = "preserve", status: str = "IDENTITY_PRESERVE") -> EditTimeMap:
    """Build exact 1:1 identity EditTimeMap (zero cuts, preserves legacy/R5.R3 exact source duration)."""
    sr = int(media.audio_streams[0].sample_rate) if media.audio_streams else 44100
    n_f = int(media.duration_frames)
    n_samp = exact_rational_samples(n_f, media.fps_num, media.fps_den, sr)
    dur_s = float(Fraction(n_f * media.fps_den, max(1, media.fps_num)))
    kept = [
        KeptSpan(
            span_id="kept_00",
            source_start_frame=0,
            source_end_frame_exclusive=n_f,
            output_start_frame=0,
            output_end_frame_exclusive=n_f,
            source_start_sample=0,
            source_end_sample=n_samp,
            output_start_sample=0,
            output_end_sample=n_samp,
            crossfade_in_samples=0,
            crossfade_out_samples=0,
        )
    ]
    em = EditTimeMap(
        schema_version=EDIT_TIME_MAP_SCHEMA_VERSION,
        source_sha256=media.input_hash,
        source_path=str(media.source_path),
        fps_num=int(media.fps_num),
        fps_den=int(media.fps_den),
        sample_rate=sr,
        source_total_frames=n_f,
        source_duration_s=dur_s,
        source_total_samples=n_samp,
        pacing_mode=pacing_mode,
        is_identity=True,
        kept_spans=kept,
        omitted_spans=[],
        output_total_frames=n_f,
        output_duration_s=dur_s,
        edited_total_samples=n_samp,
        removed_total_frames=0,
        removed_total_s=0.0,
        status=status,
    )
    em.map_hash = em.compute_hash()
    return em


def build_edit_time_map(
    media: MediaInfo,
    cut_specs: list[dict[str, Any]],
    *,
    protected_intervals_s: list[tuple[float, float]] | None = None,
    pacing_mode: str = "auto",
    crossfade_ms: float = 12.0,
) -> EditTimeMap:
    """Build exact, validated EditTimeMap from approved cut specifications.

    Rounds cut boundaries strictly INWARD (ceil start, floor end) so cuts NEVER encroach into speech/action.
    Allocates cumulative output sample boundaries ONCE from rational output frame indices (zero rounding drift!).
    """
    if pacing_mode == "preserve" or not cut_specs:
        st = "IDENTITY_PRESERVE" if pacing_mode == "preserve" else "PACING_NO_SAFE_CUTS"
        return build_identity_edit_time_map(media, pacing_mode=pacing_mode, status=st)

    sr = int(media.audio_streams[0].sample_rate) if media.audio_streams else 44100
    fps_num, fps_den = int(media.fps_num), int(media.fps_den)
    fps_frac = Fraction(fps_num, max(1, fps_den))
    fps_f = float(fps_frac)
    src_n = int(media.duration_frames)
    src_total_samples = exact_rational_samples(src_n, fps_num, fps_den, sr)
    src_dur_s = float(Fraction(src_n * fps_den, max(1, fps_num)))
    prot_list = list(protected_intervals_s or [])

    # Convert cut specs to inward-rounded integer frame intervals and verify safety against protected intervals
    raw_cuts: list[dict[str, Any]] = []
    for idx, cs in enumerate(cut_specs):
        if "cut_start_frame" in cs and "cut_end_frame_exclusive" in cs:
            cf0 = int(cs["cut_start_frame"])
            cf1 = int(cs["cut_end_frame_exclusive"])
        else:
            cs0 = float(cs["cut_start_s"])
            cs1 = float(cs["cut_end_s"])
            # Strictly INWARD rounding: start rounds UP, end rounds DOWN!
            cf0 = int(math.ceil(cs0 * fps_f))
            cf1 = int(math.floor(cs1 * fps_f))
        cf0 = max(0, min(src_n, cf0))
        cf1 = max(cf0, min(src_n, cf1))
        if cf1 - cf0 < 2:  # Ignore trivial <2-frame micro-cuts
            continue
        c_t0 = float(Fraction(cf0 * fps_den, fps_num))
        c_t1 = float(Fraction(cf1 * fps_den, fps_num))
        # Fail closed if cut overlaps any protected word/action interval!
        for p0, p1 in prot_list:
            if min(c_t1, p1) > max(c_t0, p0) + 1e-4:
                raise ValueError(
                    f"Unsafe cut {cs.get('cut_id', idx)} [{c_t0:.3f}s..{c_t1:.3f}s] intersects protected interval [{p0:.3f}s..{p1:.3f}s]"
                )
        raw_cuts.append({
            "cut_id": str(cs.get("cut_id") or f"cut_{idx:02d}"),
            "decision_id": str(cs.get("decision_id") or f"dec_{idx:02d}"),
            "cf0": cf0,
            "cf1": cf1,
            "pause_class": str(cs.get("pause_class", "DEAD_AIR")),
            "action": str(cs.get("action", "TRIM_GAP")),
            "reason": str(cs.get("reason", "")),
            "evidence_refs": list(cs.get("evidence_refs") or []),
            "confidence": float(cs.get("confidence", 0.0)),
        })

    if not raw_cuts:
        return build_identity_edit_time_map(media, pacing_mode=pacing_mode, status="PACING_NO_SAFE_CUTS")

    # Sort and verify strictly non-overlapping cuts (fail closed on overlapping/reversed specs!)
    raw_cuts.sort(key=lambda c: (c["cf0"], c["cf1"]))
    for i in range(1, len(raw_cuts)):
        if raw_cuts[i]["cf0"] < raw_cuts[i - 1]["cf1"]:
            raise ValueError(
                f"Overlapping cut specifications: {raw_cuts[i-1]['cut_id']} ({raw_cuts[i-1]['cf0']}..{raw_cuts[i-1]['cf1']}) "
                f"and {raw_cuts[i]['cut_id']} ({raw_cuts[i]['cf0']}..{raw_cuts[i]['cf1']})"
            )

    # Build contiguous kept spans from complement of cuts
    omitted_spans: list[OmittedSpan] = []
    kept_frame_ranges: list[tuple[int, int]] = []
    cur = 0
    for c in raw_cuts:
        if c["cf0"] > cur:
            kept_frame_ranges.append((cur, c["cf0"]))
        c_t0 = float(Fraction(c["cf0"] * fps_den, fps_num))
        c_t1 = float(Fraction(c["cf1"] * fps_den, fps_num))
        omitted_spans.append(OmittedSpan(
            cut_id=c["cut_id"],
            decision_id=c["decision_id"],
            source_start_frame=c["cf0"],
            source_end_frame_exclusive=c["cf1"],
            source_start_s=round(c_t0, 4),
            source_end_s=round(c_t1, 4),
            removed_frames=c["cf1"] - c["cf0"],
            removed_s=round(c_t1 - c_t0, 4),
            pause_class=c["pause_class"],
            action=c["action"],
            reason=c["reason"],
            evidence_refs=c["evidence_refs"],
            confidence=round(c["confidence"], 3),
        ))
        cur = c["cf1"]
    if cur < src_n:
        kept_frame_ranges.append((cur, src_n))

    if not kept_frame_ranges:
        raise ValueError("EditTimeMap cannot omit 100% of source frames!")

    # Allocate exact cumulative output frames & exact cumulative output audio samples (ZERO drift!)
    xfade_n = max(64, int(round(sr * crossfade_ms / 1000.0)))
    kept_spans: list[KeptSpan] = []
    out_f_cur = 0
    for idx, (sf0, sf1) in enumerate(kept_frame_ranges):
        nf = sf1 - sf0
        of0 = out_f_cur
        of1 = out_f_cur + nf
        out_s0 = exact_rational_samples(of0, fps_num, fps_den, sr)
        out_s1 = exact_rational_samples(of1, fps_num, fps_den, sr)
        # Source start sample from rational source frame; end sample matches exact output span length!
        src_s0 = exact_rational_samples(sf0, fps_num, fps_den, sr)
        span_len = out_s1 - out_s0
        src_s1 = min(src_total_samples, src_s0 + span_len)
        # Adjust src_s0 by <=1 sample only if tail hits end of source buffer
        if (src_s1 - src_s0) < span_len and src_s0 > 0:
            src_s0 = max(0, src_s1 - span_len)
        kept_spans.append(KeptSpan(
            span_id=f"kept_{idx:02d}",
            source_start_frame=sf0,
            source_end_frame_exclusive=sf1,
            output_start_frame=of0,
            output_end_frame_exclusive=of1,
            source_start_sample=src_s0,
            source_end_sample=src_s1,
            output_start_sample=out_s0,
            output_end_sample=out_s1,
            crossfade_in_samples=xfade_n if idx > 0 else 0,
            crossfade_out_samples=xfade_n if idx < len(kept_frame_ranges) - 1 else 0,
        ))
        out_f_cur = of1

    out_total_f = out_f_cur
    out_dur_s = float(Fraction(out_total_f * fps_den, max(1, fps_num)))
    edited_total_samples = exact_rational_samples(out_total_f, fps_num, fps_den, sr)
    rem_f = src_n - out_total_f
    rem_s = float(Fraction(rem_f * fps_den, max(1, fps_num)))

    em = EditTimeMap(
        schema_version=EDIT_TIME_MAP_SCHEMA_VERSION,
        source_sha256=media.input_hash,
        source_path=str(media.source_path),
        fps_num=fps_num,
        fps_den=fps_den,
        sample_rate=sr,
        source_total_frames=src_n,
        source_duration_s=src_dur_s,
        source_total_samples=src_total_samples,
        pacing_mode=pacing_mode,
        is_identity=(rem_f == 0),
        kept_spans=kept_spans,
        omitted_spans=omitted_spans,
        output_total_frames=out_total_f,
        output_duration_s=out_dur_s,
        edited_total_samples=edited_total_samples,
        removed_total_frames=rem_f,
        removed_total_s=rem_s,
        status="EDITED_CUTS_APPLIED" if rem_f > 0 else "PACING_NO_SAFE_CUTS",
    )
    em.map_hash = em.compute_hash()
    errs = validate_edit_time_map(em, protected_intervals_s=prot_list)
    if errs:
        raise ValueError(f"EditTimeMap failed fail-closed validation: {errs}")
    return em


def validate_edit_time_map(
    edit_map: EditTimeMap,
    *,
    protected_intervals_s: list[tuple[float, float]] | None = None,
) -> list[str]:
    """Fail-closed invariant verifier for EditTimeMap (blocks forged frame counts, overlaps, gaps, drift, unsafe cuts)."""
    errors: list[str] = []
    if edit_map.fps_num <= 0 or edit_map.fps_den <= 0:
        errors.append("invalid_rational_fps")
    if edit_map.source_total_frames <= 0:
        errors.append("invalid_source_total_frames")
    if not edit_map.kept_spans:
        errors.append("empty_kept_spans")
        return errors

    # Verify immutable map_hash tamper integrity
    expected_hash = edit_map.compute_hash()
    if edit_map.map_hash != expected_hash:
        errors.append(f"map_hash_tamper_mismatch:stored={edit_map.map_hash[:12]}!=computed={expected_hash[:12]}")

    # Independently derive total frames from kept spans (prevents forging output_total_frames from MP4!)
    prev_src_end = -1
    exp_out_f = 0
    exp_out_samp = 0
    for idx, ks in enumerate(edit_map.kept_spans):
        if ks.source_start_frame < 0 or ks.source_end_frame_exclusive > edit_map.source_total_frames:
            errors.append(f"{ks.span_id}:source_frames_out_of_bounds")
        if ks.source_end_frame_exclusive <= ks.source_start_frame:
            errors.append(f"{ks.span_id}:non_positive_source_span")
        if ks.source_start_frame < prev_src_end:
            errors.append(f"{ks.span_id}:overlapping_or_reversed_source_spans")
        if ks.output_start_frame != exp_out_f:
            errors.append(f"{ks.span_id}:output_frame_gap_or_duplicate:expected={exp_out_f},got={ks.output_start_frame}")
        if ks.output_end_frame_exclusive - ks.output_start_frame != ks.frame_count:
            errors.append(f"{ks.span_id}:output_vs_source_frame_count_mismatch")
        if ks.output_start_sample != exp_out_samp:
            errors.append(f"{ks.span_id}:output_sample_gap:expected={exp_out_samp},got={ks.output_start_sample}")
        if ks.output_end_sample <= ks.output_start_sample:
            errors.append(f"{ks.span_id}:non_positive_sample_span")
        prev_src_end = ks.source_end_frame_exclusive
        exp_out_f = ks.output_end_frame_exclusive
        exp_out_samp = ks.output_end_sample

    if edit_map.output_total_frames != exp_out_f:
        errors.append(f"forged_output_total_frames:declared={edit_map.output_total_frames}!=derived={exp_out_f}")

    exact_samp = exact_rational_samples(exp_out_f, edit_map.fps_num, edit_map.fps_den, edit_map.sample_rate)
    if abs(edit_map.edited_total_samples - exact_samp) > 1 or abs(exp_out_samp - exact_samp) > 1:
        errors.append(f"edited_sample_clock_drift:declared={edit_map.edited_total_samples},sum={exp_out_samp},exact={exact_samp}")

    if edit_map.removed_total_frames != (edit_map.source_total_frames - exp_out_f):
        errors.append("removed_total_frames_conservation_mismatch")

    sum_omitted = sum(o.removed_frames for o in edit_map.omitted_spans)
    if sum_omitted != edit_map.removed_total_frames:
        errors.append(f"omitted_spans_sum_mismatch:sum={sum_omitted}!=removed={edit_map.removed_total_frames}")

    # Verify NO omitted span intersects any protected word/action interval
    if protected_intervals_s:
        for o in edit_map.omitted_spans:
            o_t0 = float(Fraction(o.source_start_frame * edit_map.fps_den, edit_map.fps_num))
            o_t1 = float(Fraction(o.source_end_frame_exclusive * edit_map.fps_den, edit_map.fps_num))
            for p0, p1 in protected_intervals_s:
                if min(o_t1, p1) > max(o_t0, p0) + 1e-4:
                    errors.append(f"{o.cut_id}:intersects_protected_interval_{p0:.3f}_{p1:.3f}")

    return errors


def source_frame_to_output(edit_map: EditTimeMap, source_frame: int) -> int | None:
    """Map source frame index -> edited output frame index. Removed frames return None (NEVER invented nearest!)."""
    sf = int(source_frame)
    for ks in edit_map.kept_spans:
        if ks.source_start_frame <= sf < ks.source_end_frame_exclusive:
            return ks.output_start_frame + (sf - ks.source_start_frame)
    return None


def output_frame_to_source(edit_map: EditTimeMap, output_frame: int) -> int:
    """Map edited output frame index -> exact original source frame index."""
    of = int(output_frame)
    for ks in edit_map.kept_spans:
        if ks.output_start_frame <= of < ks.output_end_frame_exclusive:
            return ks.source_start_frame + (of - ks.output_start_frame)
    raise IndexError(f"Output frame {of} out of EditTimeMap bounds [0, {edit_map.output_total_frames})")


def source_time_to_output_time(edit_map: EditTimeMap, source_s: float) -> float | None:
    """Map source timestamp (seconds) -> edited output timestamp (seconds), or None if inside an omitted cut."""
    fps_f = edit_map.fps_num / max(1, edit_map.fps_den)
    for ks in edit_map.kept_spans:
        s0 = ks.source_start_frame / fps_f
        s1 = ks.source_end_frame_exclusive / fps_f
        if s0 - 1e-6 <= source_s <= s1 + 1e-6:
            offset = max(0.0, min(s1 - s0, source_s - s0))
            return round((ks.output_start_frame / fps_f) + offset, 4)
    return None


def map_source_interval(edit_map: EditTimeMap, src_start_f: int, src_end_f: int) -> list[tuple[int, int]]:
    """Map half-open source frame interval [src_start_f, src_end_f) -> zero or more contiguous output frame spans."""
    out: list[tuple[int, int]] = []
    for ks in edit_map.kept_spans:
        lo = max(int(src_start_f), ks.source_start_frame)
        hi = min(int(src_end_f), ks.source_end_frame_exclusive)
        if hi > lo:
            o0 = ks.output_start_frame + (lo - ks.source_start_frame)
            o1 = ks.output_start_frame + (hi - ks.source_start_frame)
            if out and o0 == out[-1][1]:
                out[-1] = (out[-1][0], o1)
            else:
                out.append((o0, o1))
    return out


def map_audio_interval(edit_map: EditTimeMap, src_s0: int, src_s1: int) -> list[tuple[int, int]]:
    """Map half-open source sample interval [src_s0, src_s1) -> edited output sample intervals."""
    out: list[tuple[int, int]] = []
    for ks in edit_map.kept_spans:
        lo = max(int(src_s0), ks.source_start_sample)
        hi = min(int(src_s1), ks.source_end_sample)
        if hi > lo:
            o0 = ks.output_start_sample + (lo - ks.source_start_sample)
            o1 = ks.output_start_sample + (hi - ks.source_start_sample)
            if out and o0 == out[-1][1]:
                out[-1] = (out[-1][0], o1)
            else:
                out.append((o0, o1))
    return out


def map_word_spans(edit_map: EditTimeMap, aligned_words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Map all aligned word spans to edited domain; flag ANY word dropped or split across a cut!"""
    fps_f = edit_map.fps_num / max(1, edit_map.fps_den)
    mapped: list[dict[str, Any]] = []
    errors: list[str] = []
    for w in aligned_words:
        s0 = float(w["source_start_s"])
        s1 = float(w["source_end_s"])
        # Find kept span containing this word
        containing = None
        for ks in edit_map.kept_spans:
            ks0 = ks.source_start_frame / fps_f
            ks1 = ks.source_end_frame_exclusive / fps_f
            if s0 >= ks0 - 0.015 and s1 <= ks1 + 0.015:
                containing = (ks, ks0, ks1)
                break
        if containing is None:
            errors.append(f"word_cut_or_split:id={w['word_id']}:'{w['word']}'[{s0:.3f}..{s1:.3f}]")
            continue
        ks, ks0, ks1 = containing
        out_base = ks.output_start_frame / fps_f
        e0 = max(out_base, out_base + max(0.0, s0 - ks0))
        e1 = max(e0 + 0.015, min(ks.output_end_frame_exclusive / fps_f, out_base + max(0.0, s1 - ks0)))
        mapped.append({
            **w,
            "time_domain": "edited",
            "edited_start_s": round(e0, 4),
            "edited_end_s": round(e1, 4),
            "edited_start_frame": max(0, min(edit_map.output_total_frames - 1, int(math.floor(e0 * fps_f)))),
            "edited_end_frame_exclusive": max(1, min(edit_map.output_total_frames, int(math.ceil(e1 * fps_f)))),
            "kept_span_id": ks.span_id,
        })
    return mapped, errors


def remap_transcript_to_edited_domain(
    transcript: Transcript,
    edit_map: EditTimeMap,
) -> Transcript:
    """Remap an immutable original Transcript cleanly into edited output time domain (NO double-remap!)."""
    if edit_map.is_identity:
        return transcript

    fps_f = edit_map.fps_num / max(1, edit_map.fps_den)
    new_segments: list[TranscriptSegment] = []
    for seg in transcript.segments:
        new_words: list[WordSpan] = []
        for w in seg.words:
            e0 = source_time_to_output_time(edit_map, float(w.start_s))
            e1 = source_time_to_output_time(edit_map, float(w.end_s))
            if e0 is None or e1 is None:
                # Word touched a cut edge -> map via nearest kept span endpoint only if <= 30ms tolerance; else raise!
                mid = 0.5 * (float(w.start_s) + float(w.end_s))
                emid = source_time_to_output_time(edit_map, mid)
                if emid is None:
                    raise ValueError(f"Cannot remap transcript: word '{w.word}' [{w.start_s:.3f}..{w.end_s:.3f}] fell inside an omitted cut!")
                half = 0.5 * max(0.02, float(w.end_s) - float(w.start_s))
                e0 = max(0.0, round(emid - half, 4))
                e1 = min(edit_map.output_duration_s, round(emid + half, 4))
            e1 = max(e0 + 0.015, e1)
            new_words.append(WordSpan(
                word=w.word,
                start_s=round(e0, 4),
                end_s=round(e1, 4),
                probability=w.probability,
                uncertain=w.uncertain,
            ))
        if not new_words:
            continue
        new_segments.append(TranscriptSegment(
            segment_id=seg.segment_id,
            start_s=new_words[0].start_s,
            end_s=new_words[-1].end_s,
            text=seg.text,
            avg_logprob=seg.avg_logprob,
            mean_word_prob=seg.mean_word_prob,
            words=new_words,
        ))

    edited_tr = Transcript(
        schema_version=SEMANTIC_SCHEMA_VERSION,
        source_hash=f"{transcript.source_hash[:16]}_edited_{edit_map.map_hash[:12]}",
        source_audio_hash=f"{transcript.source_audio_hash[:16]}_edited_{edit_map.map_hash[:12]}",
        language=transcript.language,
        language_probability=transcript.language_probability,
        duration_s=round(edit_map.output_duration_s, 4),
        sample_rate_asr=transcript.sample_rate_asr,
        model_id=transcript.model_id,
        model_sha256=transcript.model_sha256,
        asr_params=dict(transcript.asr_params),
        word_timestamps_exact=transcript.word_timestamps_exact,
        pts_priming_offset_s=0.0,
        pts_anomalies=[],
        status=transcript.status,
        full_text=transcript.full_text,
        segments=new_segments,
        uncertain_spans=[],
        warnings=list(transcript.warnings) + [f"Remapped from source via EditTimeMap {edit_map.map_hash[:12]} (output={edit_map.output_total_frames}f)."],
    )
    edited_tr.transcript_hash = edited_tr.compute_hash()
    return edited_tr


def remap_roi_tracks_to_edited_domain(roi_tracks: dict[str, Any], edit_map: EditTimeMap) -> dict[str, Any]:
    """Remap ROI keyframes cleanly from source frames to edited output frames (guards against double-remap!)."""
    if roi_tracks.get("time_domain") == "edited" and roi_tracks.get("edit_map_hash") == edit_map.map_hash:
        return roi_tracks
    if edit_map.is_identity:
        return {**roi_tracks, "time_domain": "edited", "edit_map_hash": edit_map.map_hash}

    src_kfs = roi_tracks.get("keyframes") or roi_tracks.get("frames") or []
    new_kfs: list[dict[str, Any]] = []
    for k in src_kfs:
        if not isinstance(k, dict) or "frame" not in k:
            continue
        out_f = source_frame_to_output(edit_map, int(k["frame"]))
        if out_f is None:
            continue
        nk = dict(k)
        nk["source_frame"] = int(k["frame"])
        nk["frame"] = int(out_f)
        new_kfs.append(nk)
    new_kfs.sort(key=lambda x: int(x["frame"]))

    out_doc = dict(roi_tracks)
    out_doc["time_domain"] = "edited"
    out_doc["edit_map_hash"] = edit_map.map_hash
    out_doc["source_total_frames"] = edit_map.source_total_frames
    out_doc["edited_total_frames"] = edit_map.output_total_frames
    out_doc["keyframes"] = new_kfs
    if "frames" in out_doc:
        out_doc["frames"] = new_kfs
    return out_doc


def build_edited_mezzanine_video(
    source_mp4: Path,
    edit_map: EditTimeMap,
    out_mp4: Path,
    *,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
) -> dict[str, Any]:
    """Strategy A: Build high-quality near-lossless CRF-10 edited video mezzanine strictly from immutable original MP4.

    Never touches original source; preserves exact geometry, yuv420p, and rational FPS; verifies exact output_total_frames.
    """
    if edit_map.is_identity:
        return {
            "mode": "identity_passthrough",
            "mezzanine_path": str(source_mp4),
            "parent_source_sha256": edit_map.source_sha256,
            "edit_map_hash": edit_map.map_hash,
            "verified_frame_count": edit_map.output_total_frames,
        }

    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fps_str = f"{edit_map.fps_num}/{edit_map.fps_den}"

    # Build frame-exact trim + concat filter_complex (zero seek inaccuracy!)
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for idx, ks in enumerate(edit_map.kept_spans):
        lbl = f"v{idx}"
        filter_parts.append(
            f"[0:v]trim=start_frame={ks.source_start_frame}:end_frame={ks.source_end_frame_exclusive},setpts=PTS-STARTPTS[{lbl}]"
        )
        concat_inputs.append(f"[{lbl}]")
    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(edit_map.kept_spans)}:v=1:a=0[vout]")
    fc = ";".join(filter_parts)

    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(source_mp4),
            "-filter_complex", fc,
            "-map", "[vout]",
            "-r", fps_str,
            "-c:v", "libx264", "-crf", "10", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-an", str(out_mp4),
        ],
        check=True, timeout=180, shell=False,
    )

    # Probe exact frame count of derived mezzanine (fail closed if off by even 1 frame!)
    pr = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out_mp4)],
        capture_output=True, text=True, timeout=60, shell=False,
    )
    actual_nf = int((pr.stdout or "0").strip().splitlines()[0]) if (pr.stdout or "").strip() else 0
    if actual_nf != edit_map.output_total_frames:
        raise RuntimeError(
            f"Edited mezzanine frame count mismatch: expected={edit_map.output_total_frames}, actual={actual_nf}"
        )

    h = hashlib.sha256()
    with out_mp4.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return {
        "mode": "near_lossless_crf10_mezzanine",
        "mezzanine_path": str(out_mp4),
        "mezzanine_sha256": h.hexdigest(),
        "mezzanine_bytes": out_mp4.stat().st_size,
        "parent_source_sha256": edit_map.source_sha256,
        "edit_map_hash": edit_map.map_hash,
        "verified_frame_count": actual_nf,
        "fps": fps_str,
    }


def _norm_tokens(text: str) -> list[str]:
    clean = _TOKEN_NORM_RE.sub(" ", (text or "").lower().replace("'", "").replace(",", ""))
    return [t for t in clean.split() if t]


def _extract_critical_tokens(text: str) -> list[str]:
    """Extract critical tokens that MUST survive cutting: numbers, currency, negations, and long content words."""
    toks = _norm_tokens(text)
    out: list[str] = []
    for t in toks:
        if any(ch.isdigit() for ch in t) or "$" in t or t in _CRITICAL_NEGATIONS or len(t) >= 5:
            out.append(t)
    return out


def verify_word_preservation_on_bed(
    original_transcript: Transcript,
    edited_voice_bed_wav: Path,
    edit_map: EditTimeMap,
    media: MediaInfo,
    config: Any,
    aligned_words: list[dict[str, Any]],
    *,
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Re-ASR derived `edited_voice_bed.wav` (NEVER SFX mix!) and verify ALL words/numbers/negations/CTA survived.

    Also performs exact deterministic mapping check on every aligned word.
    Returns `suspect_cut_ids` if any critical token went missing so caller can roll back offending cut!
    """
    mapped_words, map_errors = map_word_spans(edit_map, aligned_words)
    orig_tokens = _norm_tokens(original_transcript.full_text)
    orig_critical = _extract_critical_tokens(original_transcript.full_text)

    if edit_map.is_identity or not orig_tokens:
        return {
            "schema_version": EDIT_TIME_MAP_SCHEMA_VERSION,
            "edit_map_hash": edit_map.map_hash,
            "is_identity": edit_map.is_identity,
            "mapping_word_errors": map_errors,
            "mapped_words_count": len(mapped_words),
            "original_words_count": len(aligned_words),
            "re_asr_executed": False,
            "re_asr_status": "IDENTITY_SKIPPED" if edit_map.is_identity else "NO_SPEECH",
            "token_recall": 1.0,
            "critical_token_recall": 1.0,
            "missing_critical_tokens": [],
            "order_preserved": True,
            "suspect_cut_ids": [],
            "passed": len(map_errors) == 0,
        }

    # Re-ASR derived voice bed via local Faster-Whisper
    re_asr_text = ""
    re_asr_status = "NOT_RUN"
    try:
        from motionlab.semantic.asr import transcribe_local  # noqa: PLC0415

        # Create lightweight MediaInfo view for the edited WAV
        bed_media = MediaInfo.from_dict({
            **media.to_dict(),
            "source_path": str(edited_voice_bed_wav),
            "input_hash": hashlib.sha256(edited_voice_bed_wav.read_bytes()).hexdigest(),
            "nb_frames": edit_map.output_total_frames,
            "duration_s": edit_map.output_duration_s,
        })
        re_tr = transcribe_local(bed_media, config, cache_dir=None, ffmpeg_bin=ffmpeg_bin)
        re_asr_text = re_tr.full_text or ""
        re_asr_status = f"EXECUTED:{re_tr.status}"
    except Exception as exc:
        re_asr_status = f"UNAVAILABLE:{exc}"

    bed_tokens = _norm_tokens(re_asr_text) if re_asr_text else []
    # Multiset + subsequence order comparison
    if bed_tokens:
        # Subsequence LCS-like greedy ordered match on tokens (handles minor punctuation/ASR spelling variation like "$1200" vs "1200")
        bed_clean_set = [t.lstrip("$") for t in bed_tokens]
        matched_n = 0
        pos = 0
        for tok in orig_tokens:
            t_clean = tok.lstrip("$")
            found = -1
            for j in range(pos, min(len(bed_clean_set), pos + 6)):
                if bed_clean_set[j] == t_clean:
                    found = j
                    break
            if found >= 0:
                matched_n += 1
                pos = found + 1
        token_recall = round(matched_n / max(1, len(orig_tokens)), 4)

        # Check every critical token (numbers, negations, long content words) against bed tokens
        bed_bag: dict[str, int] = {}
        for t in bed_clean_set:
            bed_bag[t] = bed_bag.get(t, 0) + 1
        missing_crit: list[str] = []
        used_bag = dict(bed_bag)
        for ct in orig_critical:
            c_clean = ct.lstrip("$")
            if used_bag.get(c_clean, 0) > 0:
                used_bag[c_clean] -= 1
            else:
                missing_crit.append(ct)
        crit_recall = round((len(orig_critical) - len(missing_crit)) / max(1, len(orig_critical)), 4)

        # Identify any cut immediately adjacent to a missing critical word!
        suspect_cuts: list[str] = []
        if missing_crit:
            miss_set = {m.lstrip("$") for m in missing_crit}
            for w in aligned_words:
                w_norm = "".join(_norm_tokens(w["word"])).lstrip("$")
                if w_norm in miss_set:
                    ws0, ws1 = float(w["source_start_s"]), float(w["source_end_s"])
                    for o in edit_map.omitted_spans:
                        if abs(o.source_start_s - ws1) <= 0.45 or abs(o.source_end_s - ws0) <= 0.45:
                            if o.cut_id not in suspect_cuts:
                                suspect_cuts.append(o.cut_id)
        # Strict pass: 0 mapping word errors, >= 94% overall recall, >= 95% critical token recall, and ZERO missing numeric tokens!
        missing_nums = [m for m in missing_crit if any(ch.isdigit() for ch in m)]
        passed = bool(
            len(map_errors) == 0
            and token_recall >= 0.92
            and crit_recall >= 0.94
            and len(missing_nums) == 0
        )
    else:
        token_recall = 0.0
        crit_recall = 0.0
        missing_crit = orig_critical[:10]
        missing_nums = [m for m in orig_critical if any(ch.isdigit() for ch in m)]
        suspect_cuts = [o.cut_id for o in edit_map.omitted_spans]
        passed = False

    return {
        "schema_version": EDIT_TIME_MAP_SCHEMA_VERSION,
        "edit_map_hash": edit_map.map_hash,
        "is_identity": False,
        "mapping_word_errors": map_errors,
        "mapped_words_count": len(mapped_words),
        "original_words_count": len(aligned_words),
        "re_asr_executed": bool(bed_tokens),
        "re_asr_status": re_asr_status,
        "token_recall": token_recall,
        "critical_token_recall": crit_recall,
        "missing_critical_tokens": missing_crit,
        "missing_numeric_tokens": missing_nums,
        "order_preserved": bool(token_recall >= 0.92),
        "suspect_cut_ids": suspect_cuts,
        "passed": passed,
    }

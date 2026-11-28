"""Phase A1 / D08 (ML-01.R5.R5): Generic Domain-Agnostic Temporal Fact & State Ledger.

Enforces temporal narrative consistency across ALL domains (finance/budgeting, calendar/dates,
tutorials/step sequences, gameplay scores, historical chronology, product demos, and explicit recaps):
  - Rejects premature reveals (e.g. showing final state before its causal step executes).
  - Rejects silent state rollbacks / replays (e.g. R5.R4 defect D08 where payoff replayed
    $1,200 -> $400 -> $200 -> $100 AFTER the viewer had already reached $100!).
  - Allows deliberate flashbacks / recaps ONLY when explicitly marked `is_recap=True` with a
    valid `recap_label` and matching prior state reference.
  - Validates weekday/date ordering (e.g. Tuesday due date preceding Friday payday) and
    monotonic tutorial/process step numbers without finance-only assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.schemas import canonical_hash, sanitize_native

TEMPORAL_LEDGER_SCHEMA_VERSION = "1.0.0"

_WEEKDAY_ORDER = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


@dataclass
class TemporalEntity:
    entity_id: str
    domain: str  # "calendar" | "numeric" | "tutorial_step" | "score" | "history" | "product" | "general"
    label: str
    unit: str = ""
    monotonic_direction: str = "none"  # "increasing" | "decreasing" | "ordered_weekday" | "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class TemporalStateEvent:
    event_id: str
    entity_id: str
    scene_id: str
    start_frame: int
    end_frame_exclusive: int
    value: Any  # int | float | str
    display_text: str = ""
    unit: str = ""
    transition_kind: str = "set"  # "initial" | "delta" | "set" | "resolve" | "recap"
    depends_on_event_id: str | None = None
    is_recap: bool = False
    recap_label: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native(asdict(self))


@dataclass
class TemporalFactLedger:
    schema_version: str = TEMPORAL_LEDGER_SCHEMA_VERSION
    time_domain: str = "edited"  # "source" | "edited"
    fps_num: int = 30
    fps_den: int = 1
    entities: dict[str, TemporalEntity] = field(default_factory=dict)
    events: list[TemporalStateEvent] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    valid: bool = True
    ledger_hash: str = ""

    def add_entity(self, entity: TemporalEntity) -> None:
        self.entities[entity.entity_id] = entity

    def add_event(self, event: TemporalStateEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        return sanitize_native({
            "schema_version": self.schema_version,
            "time_domain": self.time_domain,
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "events": [e.to_dict() for e in self.events],
            "validation_errors": list(self.validation_errors),
            "valid": bool(self.valid),
            "ledger_hash": self.ledger_hash,
        })

    def compute_hash(self) -> str:
        d = self.to_dict()
        d.pop("ledger_hash", None)
        return canonical_hash(d)


def _to_numeric(val: Any) -> float | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).strip().replace(",", "").lstrip("$€£¥")
        return float(s)
    except Exception:
        return None


def _to_weekday_idx(val: Any) -> int | None:
    s = str(val or "").strip().lower()
    return _WEEKDAY_ORDER.get(s)


def validate_temporal_fact_ledger(ledger: TemporalFactLedger) -> list[str]:
    """Fail-closed temporal semantic checker rejecting silent rollbacks, premature states & contradictions."""
    errors: list[str] = []
    events_by_id: dict[str, TemporalStateEvent] = {}
    for ev in ledger.events:
        if ev.event_id in events_by_id:
            errors.append(f"duplicate_event_id:{ev.event_id}")
        events_by_id[ev.event_id] = ev
        if ev.entity_id not in ledger.entities:
            errors.append(f"unknown_entity:{ev.event_id}:{ev.entity_id}")
        if ev.end_frame_exclusive <= ev.start_frame:
            errors.append(f"non_positive_event_span:{ev.event_id}:{ev.start_frame}..{ev.end_frame_exclusive}")
        if ev.is_recap and not ev.recap_label.strip():
            errors.append(f"recap_missing_explicit_label:{ev.event_id}")

    # Dependency causal ordering: every dependency MUST begin before or at the dependent event
    for ev in ledger.events:
        if ev.depends_on_event_id:
            dep = events_by_id.get(ev.depends_on_event_id)
            if dep is None:
                errors.append(f"missing_dependency:{ev.event_id}->{ev.depends_on_event_id}")
            elif dep.start_frame > ev.start_frame:
                errors.append(
                    f"premature_state_before_dependency:{ev.event_id}@f{ev.start_frame}<{dep.event_id}@f{dep.start_frame}"
                )

    # Group events per entity ordered chronologically by start_frame
    by_entity: dict[str, list[TemporalStateEvent]] = {}
    for ev in ledger.events:
        by_entity.setdefault(ev.entity_id, []).append(ev)

    for eid, evs in by_entity.items():
        ent = ledger.entities.get(eid)
        if ent is None:
            continue
        ordered = sorted(evs, key=lambda x: (x.start_frame, x.end_frame_exclusive))
        seen_values: list[Any] = []
        reached_resolved = False
        prev_ev: TemporalStateEvent | None = None

        for ev in ordered:
            # Check conflicting simultaneous values for the same entity
            if prev_ev is not None and ev.start_frame < prev_ev.end_frame_exclusive:
                if ev.value != prev_ev.value and not ev.is_recap:
                    errors.append(
                        f"simultaneous_conflicting_state:{eid}:{prev_ev.event_id}({prev_ev.value}) vs {ev.event_id}({ev.value})"
                    )

            # D08 Guard: Once an entity has reached a terminal "resolve" state, silently replaying
            # an earlier pre-resolution value without explicit `is_recap=True` is FORBIDDEN!
            if reached_resolved and not ev.is_recap and ev.transition_kind != "recap":
                if prev_ev is not None and ev.value != prev_ev.value:
                    errors.append(
                        f"silent_post_resolution_rollback:{eid}:{ev.event_id}({ev.value})_after_resolved({prev_ev.value})"
                    )

            # Domain monotonic checks (non-recap only)
            if not ev.is_recap and prev_ev is not None and not prev_ev.is_recap:
                if ent.monotonic_direction == "decreasing":
                    v_prev = _to_numeric(prev_ev.value)
                    v_cur = _to_numeric(ev.value)
                    if v_prev is not None and v_cur is not None and v_cur > v_prev + 1e-6:
                        errors.append(
                            f"temporal_rollback_violation:{eid}:{ev.event_id}({v_cur})>prev({v_prev})_without_recap"
                        )
                elif ent.monotonic_direction == "increasing":
                    v_prev = _to_numeric(prev_ev.value)
                    v_cur = _to_numeric(ev.value)
                    if v_prev is not None and v_cur is not None and v_cur < v_prev - 1e-6:
                        errors.append(
                            f"temporal_regression_violation:{eid}:{ev.event_id}({v_cur})<prev({v_prev})_without_recap"
                        )
                elif ent.monotonic_direction == "ordered_weekday":
                    d_prev = _to_weekday_idx(prev_ev.value)
                    d_cur = _to_weekday_idx(ev.value)
                    if d_prev is not None and d_cur is not None and d_cur < d_prev:
                        errors.append(
                            f"weekday_order_violation:{eid}:{ev.event_id}({ev.value})<prev({prev_ev.value})"
                        )

            # Recap events MUST reference an actually previously observed value
            if ev.is_recap and seen_values and ev.value not in seen_values:
                errors.append(f"ungrounded_recap_value:{eid}:{ev.event_id}({ev.value})")

            if ev.transition_kind == "resolve":
                reached_resolved = True
            seen_values.append(ev.value)
            prev_ev = ev

    ledger.validation_errors = errors
    ledger.valid = len(errors) == 0
    ledger.ledger_hash = ledger.compute_hash()
    return errors


def build_temporal_ledger_from_story_and_scenes(
    *,
    scenes: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]] | None = None,
    fps_num: int = 30,
    fps_den: int = 1,
    time_domain: str = "edited",
) -> TemporalFactLedger:
    """Build and validate a TemporalFactLedger automatically from scenes & spoken weekday/numeric events."""
    ledger = TemporalFactLedger(time_domain=time_domain, fps_num=fps_num, fps_den=fps_den)

    # 1. Track spoken calendar/weekday progression if present in transcript words
    words = transcript_words or []
    fps_f = fps_num / max(1, fps_den)
    cal_events: list[tuple[int, str, str]] = []
    for w in words:
        tok = str(w.get("word", "")).strip().lower().strip(".,?!;:\"'")
        if tok in _WEEKDAY_ORDER:
            s_t = float(w.get("output_start_s", w.get("start_s", 0.0)))
            f0 = int(round(s_t * fps_f))
            cal_events.append((f0, tok.capitalize(), f"word:{w.get('word_id', 0)}"))
    if cal_events:
        ledger.add_entity(TemporalEntity(
            entity_id="calendar_timeline",
            domain="calendar",
            label="Spoken Weekday Progression",
            unit="weekday",
            monotonic_direction="ordered_weekday",
        ))
        seen_day: set[str] = set()
        for idx, (f0, day_lbl, ev_ref) in enumerate(cal_events):
            if day_lbl in seen_day:
                continue
            seen_day.add(day_lbl)
            ledger.add_event(TemporalStateEvent(
                event_id=f"cal_{idx:02d}_{day_lbl.lower()}",
                entity_id="calendar_timeline",
                scene_id="transcript_observed",
                start_frame=f0,
                end_frame_exclusive=f0 + max(1, int(round(fps_f * 0.5))),
                value=day_lbl,
                display_text=day_lbl.upper()[:3],
                unit="weekday",
                transition_kind="initial" if idx == 0 else "set",
                evidence_refs=[ev_ref],
            ))

    # 2. Track scene-emitted numeric or stepped state transitions
    has_numeric = False
    for s_idx, sc in enumerate(scenes):
        st = sc.get("state_transition") or {}
        if not isinstance(st, dict) or not st:
            continue
        eid = str(st.get("entity_id", "primary_state"))
        if eid not in ledger.entities:
            ledger.add_entity(TemporalEntity(
                entity_id=eid,
                domain=str(st.get("domain", "general")),
                label=str(st.get("label", eid)),
                unit=str(st.get("unit", "")),
                monotonic_direction=str(st.get("monotonic_direction", "none")),
            ))
        ledger.add_event(TemporalStateEvent(
            event_id=str(st.get("event_id", f"st_{s_idx:02d}")),
            entity_id=eid,
            scene_id=str(sc.get("scene_id", f"sc_{s_idx:02d}")),
            start_frame=int(sc.get("start_frame", 0)),
            end_frame_exclusive=max(int(sc.get("start_frame", 0)) + 1, int(sc.get("end_frame_exclusive", 1))),
            value=st.get("value"),
            display_text=str(st.get("display_text", "")),
            unit=str(st.get("unit", "")),
            transition_kind=str(st.get("transition_kind", "set")),
            depends_on_event_id=st.get("depends_on_event_id"),
            is_recap=bool(st.get("is_recap", False)),
            recap_label=str(st.get("recap_label", "")),
            evidence_refs=list(st.get("evidence_refs", [])),
        ))
        has_numeric = True

    # 3. Always track narrative beat progression monotonically
    if scenes:
        ledger.add_entity(TemporalEntity(
            entity_id="narrative_beat_order",
            domain="general",
            label="Editorial Story Arc Progression",
            unit="scene_index",
            monotonic_direction="increasing",
        ))
        for idx, sc in enumerate(scenes):
            sf = int(sc.get("start_frame", 0))
            ef = max(sf + 1, int(sc.get("end_frame_exclusive", sf + 1)))
            ledger.add_event(TemporalStateEvent(
                event_id=f"arc_{idx:02d}_{sc.get('scene_id', idx)}",
                entity_id="narrative_beat_order",
                scene_id=str(sc.get("scene_id", f"sc_{idx:02d}")),
                start_frame=sf,
                end_frame_exclusive=ef,
                value=idx,
                display_text=str(sc.get("narrative_role", sc.get("role", "beat"))),
                transition_kind="resolve" if idx == len(scenes) - 1 else ("initial" if idx == 0 else "set"),
                evidence_refs=list(sc.get("evidence_ids", [])),
            ))

    validate_temporal_fact_ledger(ledger)
    return ledger

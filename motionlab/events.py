"""Phase B7 — Deterministic candidate event builder from local media signals.

Produces sorted, deduplicated, frame-indexed candidate events bounded per minute:
  - scene_boundary
  - sustained_static
  - audio_onset
  - energy_rise
  - motion_change
  - safe_hold

Stores evidence metrics only (never full frames).
"""

from __future__ import annotations

from motionlab.schemas import AnalysisResult, CandidateEvent, MediaInfo


def build_candidates(
    analysis: AnalysisResult,
    media: MediaInfo,
    max_per_minute: int = 40,
) -> list[CandidateEvent]:
    """Convert raw signal analysis into sorted, merged, bounded candidate events."""
    fps = media.fps_float
    dur_frames = analysis.duration_frames
    if dur_frames <= 1:
        return []

    raw: list[CandidateEvent] = []
    idx_counter = 0

    def next_id(prefix: str) -> str:
        nonlocal idx_counter
        idx_counter += 1
        return f"cand_{prefix}_{idx_counter:03d}"

    # 1. Scene boundaries (detected cuts not protected as existing gradual transitions)
    for cut in analysis.cuts:
        if cut.protected_existing_transition:
            continue
        # Avoid cuts too close to start/end (within 0.4s)
        margin = max(2, int(round(fps * 0.4)))
        if cut.frame < margin or cut.frame > dur_frames - margin:
            continue
        half_win = max(2, int(round(fps * 0.25)))
        raw.append(
            CandidateEvent(
                id=next_id("cut"),
                frame=cut.frame,
                end_frame_exclusive=min(dur_frames, cut.frame + half_win),
                event_type="scene_boundary",
                confidence=cut.confidence,
                reason=f"detected_scene_cut_hist={cut.hist_diff:.2f}",
                metrics={"hist_diff": cut.hist_diff, "luma_diff": cut.luma_diff},
            )
        )

    # 2. Sustained static windows (contiguous near-static windows >= 2.0s)
    static_start: int | None = None
    static_end: int = 0
    static_stabs: list[float] = []
    for mw in analysis.motion_windows:
        if mw.motion_class == "near_static":
            if static_start is None:
                static_start = mw.start_frame
                static_stabs = []
            static_end = mw.end_frame_exclusive
            static_stabs.append(mw.stability_score)
        else:
            if static_start is not None and (static_end - static_start) >= int(round(fps * 1.8)):
                raw.append(
                    CandidateEvent(
                        id=next_id("static"),
                        frame=static_start,
                        end_frame_exclusive=static_end,
                        event_type="sustained_static",
                        confidence=min(0.95, sum(static_stabs) / max(1, len(static_stabs))),
                        reason="sustained_low_motion_window",
                        metrics={
                            "duration_s": (static_end - static_start) / max(fps, 1.0),
                            "stability": sum(static_stabs) / max(1, len(static_stabs)),
                        },
                    )
                )
            static_start = None
    if static_start is not None and (static_end - static_start) >= int(round(fps * 1.8)):
        raw.append(
            CandidateEvent(
                id=next_id("static"),
                frame=static_start,
                end_frame_exclusive=static_end,
                event_type="sustained_static",
                confidence=min(0.95, sum(static_stabs) / max(1, len(static_stabs))),
                reason="sustained_low_motion_window",
                metrics={
                    "duration_s": (static_end - static_start) / max(fps, 1.0),
                    "stability": sum(static_stabs) / max(1, len(static_stabs)),
                },
            )
        )

    # 3. Motion changes (transitions from low->moderate or moderate->low)
    for i in range(1, len(analysis.motion_windows)):
        prev_mw = analysis.motion_windows[i - 1]
        cur_mw = analysis.motion_windows[i]
        delta = cur_mw.motion_score - prev_mw.motion_score
        if abs(delta) >= 0.14:
            raw.append(
                CandidateEvent(
                    id=next_id("motchg"),
                    frame=cur_mw.start_frame,
                    end_frame_exclusive=cur_mw.end_frame_exclusive,
                    event_type="motion_change",
                    confidence=min(0.9, 0.45 + abs(delta)),
                    reason="measured_motion_delta",
                    metrics={"motion_delta": delta, "motion_score": cur_mw.motion_score},
                )
            )

    # 4. Safe holds (moderate/calm windows suitable for gentle framing hold / grading / edge accents)
    for mw in analysis.motion_windows:
        if mw.motion_class in ("near_static", "moderate") and mw.stability_score >= 0.55:
            raw.append(
                CandidateEvent(
                    id=next_id("hold"),
                    frame=mw.start_frame,
                    end_frame_exclusive=mw.end_frame_exclusive,
                    event_type="safe_hold",
                    confidence=min(0.88, mw.stability_score),
                    reason=f"stable_window_motion={mw.motion_score:.2f}",
                    metrics={
                        "motion_score": mw.motion_score,
                        "stability": mw.stability_score,
                        "saliency_x": mw.saliency_center_x,
                        "saliency_y": mw.saliency_center_y,
                    },
                )
            )

    # 5. Acoustic onset cues (energy peaks from local PCM envelope — signal cues, NOT semantic meaning)
    for pk in analysis.energy_peaks:
        span = max(2, int(round(fps * 1.2)))
        ef = min(dur_frames, pk.frame + span)
        ev_type = "audio_onset" if pk.onset_flux >= 0.45 else "energy_rise"
        raw.append(
            CandidateEvent(
                id=next_id("audio"),
                frame=pk.frame,
                end_frame_exclusive=ef,
                event_type=ev_type,
                confidence=pk.confidence,
                reason=f"local_acoustic_cue_rms={pk.strength:.2f}",
                metrics={"rms_norm": pk.strength, "onset_flux": pk.onset_flux},
            )
        )

    # 6. Sort deterministically & merge near-duplicates of the same type within 0.45s
    raw.sort(key=lambda c: (c.frame, c.event_type, -c.confidence, c.id))
    merge_gap = max(2, int(round(fps * 0.45)))
    merged: list[CandidateEvent] = []
    for cand in raw:
        if (
            merged
            and merged[-1].event_type == cand.event_type
            and abs(cand.frame - merged[-1].frame) <= merge_gap
        ):
            # Keep higher confidence one
            if cand.confidence > merged[-1].confidence:
                merged[-1] = cand
            continue
        merged.append(cand)

    # 7. Bound candidate count per minute while guaranteeing scene_boundaries survive
    max_total = max(6, int(round(max_per_minute * max(media.duration_s / 60.0, 0.25))))
    if len(merged) > max_total:
        must_keep = [c for c in merged if c.event_type == "scene_boundary"]
        others = [c for c in merged if c.event_type != "scene_boundary"]
        others.sort(key=lambda c: (-c.confidence, c.frame, c.id))
        kept = must_keep + others[: max(0, max_total - len(must_keep))]
        kept.sort(key=lambda c: (c.frame, c.event_type, c.id))
        merged = kept

    return merged

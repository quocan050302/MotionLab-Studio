"""Phase B7: Candidate event extraction, deduplication, and deterministic sorting."""

from __future__ import annotations

from motionlab.schemas import AnalysisResult, CandidateEvent


def build_candidates(analysis: AnalysisResult, max_per_minute: int = 24) -> list[CandidateEvent]:
    """Convert raw local signal analysis into bounded, deduplicated candidate events.

    Stores only evidence metrics, never raw video frames.
    """
    fps_f = analysis.fps.as_float()
    total_frames = analysis.duration_frames
    if total_frames <= 1:
        return []

    raw: list[CandidateEvent] = []

    # 1. Scene boundaries
    for idx, cut in enumerate(analysis.cuts):
        half_w = max(3, int(round(0.25 * fps_f)))
        sf = max(0, cut.frame - half_w)
        ef = min(total_frames, cut.frame + half_w)
        if ef > sf + 2:
            raw.append(
                CandidateEvent(
                    id=f"cand_cut_{idx:03d}",
                    frame=cut.frame,
                    start_frame=sf,
                    end_frame_exclusive=ef,
                    event_type="scene_boundary",
                    confidence=cut.confidence,
                    reason=f"detected_scene_cut(hist={cut.hist_diff},luma={cut.luma_diff})",
                    evidence={"hist_diff": cut.hist_diff, "luma_diff": cut.luma_diff, "boundary_type": cut.boundary_type},
                )
            )

    # 2. Sustained static windows (great for gentle push-in / drift / layout reveals)
    static_run_start: int | None = None
    static_run_end: int = 0
    static_stabilities: list[float] = []
    static_idx = 0

    def _emit_static() -> None:
        nonlocal static_run_start, static_run_end, static_stabilities, static_idx
        if static_run_start is not None:
            dur_s = (static_run_end - static_run_start) / fps_f
            if dur_s >= 1.4:
                mid = (static_run_start + static_run_end) // 2
                mean_stab = sum(static_stabilities) / max(1, len(static_stabilities))
                raw.append(
                    CandidateEvent(
                        id=f"cand_static_{static_idx:03d}",
                        frame=mid,
                        start_frame=static_run_start,
                        end_frame_exclusive=static_run_end,
                        event_type="sustained_static",
                        confidence=round(min(0.95, 0.55 + 0.4 * mean_stab), 3),
                        reason=f"sustained_low_motion({dur_s:.2f}s,stability={mean_stab:.2f})",
                        evidence={"duration_s": round(dur_s, 3), "mean_stability": round(mean_stab, 4)},
                    )
                )
                static_idx += 1
        static_run_start = None
        static_run_end = 0
        static_stabilities = []

    for mw in analysis.motion_windows:
        if mw.near_static:
            if static_run_start is None:
                static_run_start = mw.start_frame
            static_run_end = mw.end_frame_exclusive
            static_stabilities.append(mw.stability)
        else:
            _emit_static()
    _emit_static()

    # 3. Motion changes (transitions between calm and active motion)
    for i in range(1, len(analysis.motion_windows)):
        prev_m = analysis.motion_windows[i - 1]
        curr_m = analysis.motion_windows[i]
        delta = abs(curr_m.motion_score - prev_m.motion_score)
        if delta >= 0.22:
            raw.append(
                CandidateEvent(
                    id=f"cand_mchg_{i:03d}",
                    frame=curr_m.start_frame,
                    start_frame=curr_m.start_frame,
                    end_frame_exclusive=curr_m.end_frame_exclusive,
                    event_type="motion_change",
                    confidence=round(min(0.9, 0.5 + delta), 3),
                    reason=f"motion_delta({prev_m.motion_score:.2f}->{curr_m.motion_score:.2f})",
                    evidence={"delta": round(delta, 4), "motion_class": curr_m.motion_class},
                )
            )

    # 4. Audio onsets & energy rises (acoustic cues ONLY, never semantic emphasis claims)
    for idx, pk in enumerate(analysis.energy_peaks):
        pre = max(0, pk.frame - int(round(0.18 * fps_f)))  # anticipation window ~180ms
        post = min(total_frames, pk.frame + int(round(1.6 * fps_f)))
        if post - pre < max(4, int(round(0.35 * fps_f))):
            continue
        ev_type = "audio_onset" if pk.onset_flux >= 0.15 else "energy_rise"
        raw.append(
            CandidateEvent(
                id=f"cand_aud_{idx:03d}",
                frame=pk.frame,
                start_frame=pre,
                end_frame_exclusive=post,
                event_type=ev_type,
                confidence=round(min(0.95, 0.45 + 0.5 * pk.strength), 3),
                reason=f"acoustic_cue(strength={pk.strength:.2f},rms_norm={pk.rms_norm:.2f})",
                evidence={"strength": pk.strength, "onset_flux": pk.onset_flux, "rms_norm": pk.rms_norm},
            )
        )

    # 5. Safe hold windows (calm mid-shot segments useful for gentle grade/framing accents even without audio)
    if total_frames >= int(round(2.0 * fps_f)):
        step_f = max(int(round(3.5 * fps_f)), total_frames // 5)
        hold_idx = 0
        for f in range(int(round(0.5 * fps_f)), total_frames - int(round(1.2 * fps_f)), max(1, step_f)):
            ef = min(total_frames - 2, f + int(round(2.4 * fps_f)))
            if ef > f + int(round(0.8 * fps_f)):
                raw.append(
                    CandidateEvent(
                        id=f"cand_hold_{hold_idx:03d}",
                        frame=f,
                        start_frame=f,
                        end_frame_exclusive=ef,
                        event_type="safe_hold",
                        confidence=0.60,
                        reason="paced_safe_interval",
                        evidence={"anchor_frame": f},
                    )
                )
                hold_idx += 1

    # Merge near-duplicates of the same event_type within 10 frames (keep highest confidence)
    raw.sort(key=lambda c: (c.frame, c.event_type, -c.confidence, c.id))
    deduped: list[CandidateEvent] = []
    merge_gap = max(6, int(round(0.30 * fps_f)))
    for cand in raw:
        merged = False
        for idx, prev in enumerate(deduped):
            if prev.event_type == cand.event_type and abs(prev.frame - cand.frame) <= merge_gap:
                if cand.confidence > prev.confidence:
                    deduped[idx] = cand
                merged = True
                break
        if not merged:
            deduped.append(cand)

    # Bound total candidate count per minute while always preserving scene boundaries first
    duration_min = max(0.25, (total_frames / fps_f) / 60.0)
    max_total = max(8, int(round(max_per_minute * duration_min)))
    if len(deduped) > max_total:
        boundaries = [c for c in deduped if c.event_type == "scene_boundary"]
        others = [c for c in deduped if c.event_type != "scene_boundary"]
        others.sort(key=lambda c: (-c.confidence, c.frame, c.id))
        keep_n = max(0, max_total - len(boundaries))
        deduped = boundaries + others[:keep_n]

    deduped.sort(key=lambda c: (c.frame, c.start_frame, c.event_type, c.id))
    return deduped

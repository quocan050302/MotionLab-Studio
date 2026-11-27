"""Phase 6B (ML-01.R4): Sample-Exact Rational Audio Cue Planner (Frame/Sample Clock + Cooldown + Clamping)."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from motionlab.audio.schema import AUDIO_SCHEMA_VERSION, AudioTimeline, SoundEvent
from motionlab.audio.synthesis import SFX_FAMILIES, synthesize_sfx_waveform
from motionlab.schemas import MediaInfo
from motionlab.semantic.niche import NicheStyleProfile
from motionlab.semantic.schema import SemanticDirectorPlan

# Restrained, editorial starting gain levels (dB relative to normalized -2 dBFS asset)
# Designed to sit naturally ~12-18 dB below active voice (further smoothed by speech ducking in mixer.py!)
_FAMILY_BASE_GAIN_DB: dict[str, float] = {
    "soft_whoosh": -12.5,
    "reverse_whoosh": -12.0,
    "pop_reveal": -13.0,
    "soft_impact": -11.5,
    "cash_flick": -11.5,
    "paper_rustle": -12.5,
    "envelope_slide": -12.0,
    "tick_accent": -13.5,
    "shimmer_payoff": -13.0,
    "riser_build": -13.5,
    "bass_settle": -12.0,
    "ambience_pad": -26.0,
}


def frame_to_sample_exact(frame: int, fps_num: int, fps_den: int, sample_rate: int) -> int:
    """Convert integer frame index to exact sample index using rational Fraction arithmetic (24/25/29.97/30/60 FPS)."""
    if frame <= 0:
        return 0
    fps_frac = Fraction(fps_num, max(1, fps_den))
    return int(round(Fraction(frame, 1) / fps_frac * sample_rate))


def build_audio_cues(
    media: MediaInfo,
    semantic_plan: SemanticDirectorPlan | None,
    niche_profile: NicheStyleProfile | None,
    audio_mode: str = "mix",
    sfx_enabled: bool = True,
    music_enabled: bool = False,
    seed: int = 42,
    sample_rate_override: int | None = None,
) -> tuple[AudioTimeline, dict[str, Any]]:
    """Build frame/sample-synchronized SoundEvent timeline linked to semantic beats & observed visual actions."""
    sr = sample_rate_override or (media.audio_streams[0].sample_rate if media.audio_streams else 44100)
    channels = 2 if (audio_mode == "mix" and sfx_enabled) else (media.audio_streams[0].channels if media.audio_streams else 1)
    fps_num, fps_den = media.fps.num, media.fps.den
    dur_samples = frame_to_sample_exact(media.duration_frames, fps_num, fps_den, sr)
    audio_pts = [s for s in media.pts_summary if s.get("codec_type") == "audio"]
    priming_s = float(audio_pts[0].get("priming_offset_s", 0.0)) if audio_pts else 0.0

    if audio_mode == "preserve-exact" or not sfx_enabled or semantic_plan is None or not semantic_plan.beats:
        tl = AudioTimeline(
            schema_version=AUDIO_SCHEMA_VERSION,
            source_hash=media.input_hash,
            audio_mode=audio_mode,
            sfx_enabled=bool(sfx_enabled and audio_mode == "mix"),
            music_enabled=bool(music_enabled and audio_mode == "mix"),
            sample_rate=sr,
            channels=channels,
            duration_samples=dur_samples,
            fps_num=fps_num,
            fps_den=fps_den,
            pts_priming_offset_s=round(priming_s, 6),
            events=[],
        )
        tl.timeline_hash = tl.compute_hash()
        return tl, {"used_assets": []}

    gain_offset = float(niche_profile.sfx_gain_offset_db) if niche_profile else 0.0
    events: list[SoundEvent] = []
    used_assets: dict[str, dict[str, Any]] = {}
    last_peak_by_fam: dict[str, int] = {}
    min_fam_cooldown_s = 0.85
    min_any_gap_s = 0.30
    last_any_peak = -sr
    variant_counter: dict[str, int] = {}

    for b_idx, beat in enumerate(semantic_plan.beats):
        for r_idx, cue in enumerate(beat.sfx_roles):
            fam = str(cue.get("role") or "soft_whoosh")
            if fam not in SFX_FAMILIES:
                continue
            target_f = max(1, min(media.duration_frames - 2, int(cue.get("target_frame", beat.start_frame + 4))))
            peak_samp = frame_to_sample_exact(target_f, fps_num, fps_den, sr)

            # Enforce tasteful anti-spam spacing (no overlapping cluttered hits!)
            if peak_samp - last_any_peak < int(sr * min_any_gap_s):
                continue
            if fam in last_peak_by_fam and (peak_samp - last_peak_by_fam[fam]) < int(sr * min_fam_cooldown_s):
                continue

            var = variant_counter.get(fam, 0) % 3
            variant_counter[fam] = var + 1
            wav, spec = synthesize_sfx_waveform(fam, variant=var, sr=sr, seed=seed)
            used_assets[spec.asset_id] = spec.to_dict()

            trans_samp = int(round(spec.transient_offset_s * sr))
            start_samp = peak_samp - trans_samp
            if start_samp < 0:
                start_samp = 0
                peak_samp = min(dur_samples - 1, start_samp + trans_samp)
            end_samp = start_samp + wav.size
            if end_samp > dur_samples:
                # Clamp cleanly at timeline end without ever truncating voice or extending duration!
                end_samp = dur_samples
                if end_samp - start_samp < int(sr * 0.05):
                    continue

            # Gentle alternating editorial stereo width (stays narrow & centered for phone playback!)
            pan = 0.0 if fam in ("soft_impact", "bass_settle", "shimmer_payoff") else ((-0.16, 0.16)[(b_idx + r_idx) % 2])
            gain_db = round(_FAMILY_BASE_GAIN_DB.get(fam, -13.0) + gain_offset, 2)

            events.append(SoundEvent(
                id=f"sfx_{len(events):02d}_{fam}",
                linked_visual_event_id=beat.beat_id,
                semantic_span_ids=list(beat.unit_ids),
                observed_action_id=f"beat_{b_idx:02d}_{beat.recipe_id}",
                asset_id=spec.asset_id,
                family=fam,
                variant=var,
                asset_sha256=spec.sha256,
                start_sample=start_samp,
                peak_sample=peak_samp,
                end_sample=end_samp,
                peak_frame=target_f,
                gain_db=gain_db,
                pan=pan,
                role=str(cue.get("reason") or fam),
                confidence=beat.confidence,
                reason=f"{beat.recipe_id}:{cue.get('reason', fam)}",
            ))
            last_peak_by_fam[fam] = peak_samp
            last_any_peak = peak_samp

    # Cap total cues gracefully at 14 on a 30s clip to stay restrained and cinematic
    events = events[:14]
    tl = AudioTimeline(
        schema_version=AUDIO_SCHEMA_VERSION,
        source_hash=media.input_hash,
        audio_mode=audio_mode,
        sfx_enabled=True,
        music_enabled=bool(music_enabled),
        sample_rate=sr,
        channels=channels,
        duration_samples=dur_samples,
        fps_num=fps_num,
        fps_den=fps_den,
        pts_priming_offset_s=round(priming_s, 6),
        events=events,
    )
    tl.timeline_hash = tl.compute_hash()
    return tl, {"used_assets": list(used_assets.values())}

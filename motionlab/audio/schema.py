"""Typed schemas for ML-01.R4 local sound design assets, sample-exact cues, audio timeline, and mix QA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motionlab.schemas import canonical_hash

AUDIO_SCHEMA_VERSION = "1.0.0"


@dataclass
class SfxAssetSpec:
    asset_id: str
    family: str
    variant: int
    provenance: str  # "procedural_dsp_synthesis" or verified CC0/CC-BY local asset
    license: str
    attribution: str
    sample_rate: int
    channels: int
    duration_s: float
    transient_offset_s: float
    peak_dbfs: float
    sha256: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SoundEvent:
    id: str
    linked_visual_event_id: str
    semantic_span_ids: list[str]
    observed_action_id: str
    asset_id: str
    family: str
    variant: int
    asset_sha256: str
    start_sample: int
    peak_sample: int
    end_sample: int
    peak_frame: int
    gain_db: float
    pan: float  # [-0.6, +0.6], 0.0 center
    role: str
    confidence: float
    reason: str
    fallback: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioTimeline:
    schema_version: str
    source_hash: str
    audio_mode: str  # "mix" or "preserve-exact"
    sfx_enabled: bool
    music_enabled: bool
    sample_rate: int
    channels: int
    duration_samples: int
    fps_num: int
    fps_den: int
    pts_priming_offset_s: float
    events: list[SoundEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timeline_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("timeline_hash", None)
        return canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["events"] = [e.to_dict() for e in self.events]
        return d


@dataclass
class MixMetrics:
    audio_mode: str
    sample_rate: int
    channels: int
    duration_samples: int
    original_master_packet_sha256: str
    original_master_packets: int
    source_peak_dbfs: float
    source_lufs_i: float
    sfx_stem_peak_dbfs: float
    sfx_active_events: int
    mix_preencode_peak_dbfs: float
    mix_preencode_lufs_i: float
    final_decoded_peak_dbfs: float
    final_decoded_lufs_i: float
    final_vs_ref_correlation: float
    voice_master_unity_preserved: bool
    sfx_headroom_scale_applied: float
    clipping_samples_added: int
    music_enabled: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

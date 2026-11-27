"""ML-01.R4 Local Sound Design Engine: Procedural DSP SFX, Frame/Sample Sync Cues, Master-Preserving Mixer & QA."""

from motionlab.audio.assets import inspect_foley_library
from motionlab.audio.cues import build_audio_cues
from motionlab.audio.mixer import extract_original_audio_master, mix_preserving_master, validate_audio_mix, validate_final_audio_mix
from motionlab.audio.schema import AudioTimeline, MixMetrics, SfxAssetSpec, SoundEvent
from motionlab.audio.synthesis import SFX_FAMILIES, build_sfx_asset_manifest, synthesize_sfx_variant, synthesize_sfx_waveform

__all__ = [
    "SfxAssetSpec",
    "SoundEvent",
    "AudioTimeline",
    "MixMetrics",
    "SFX_FAMILIES",
    "inspect_foley_library",
    "synthesize_sfx_variant",
    "synthesize_sfx_waveform",
    "build_sfx_asset_manifest",
    "build_audio_cues",
    "extract_original_audio_master",
    "mix_preserving_master",
    "validate_audio_mix",
    "validate_final_audio_mix",
]

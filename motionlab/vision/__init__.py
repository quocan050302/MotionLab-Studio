"""Phase 1 (ML-01.R3): Local Zero-API Subject Segmentation, ROI Tracking, Matte Refinement & Anchor Tracks."""

from motionlab.vision.anchors import build_anchor_tracks, generate_matte_debug_sheet, run_vision_pipeline
from motionlab.vision.segmentation import (
    MODNetSegmenter,
    build_foreground_alpha_webm,
    inspect_vision_model,
    refine_matte_temporally,
    segment_subject_sequence,
)
from motionlab.vision.tracking import track_action_rois

__all__ = [
    "MODNetSegmenter",
    "build_anchor_tracks",
    "build_foreground_alpha_webm",
    "generate_matte_debug_sheet",
    "inspect_vision_model",
    "refine_matte_temporally",
    "run_vision_pipeline",
    "segment_subject_sequence",
    "track_action_rois",
]

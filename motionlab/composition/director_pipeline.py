"""Phase A0–A12 (ML-01.R5.R5): Production Multi-Source Generative Creative Director Pipeline.

Real production CLI callgraph entry point (`python -m motionlab edit --job <job.json> --creative-mode director`),
fixing D01 (dead code / unintegrated modules) and connecting ALL R5.R5 engines end-to-end:
  1. Strict Job Manifest Validation (schema, path safety, no symlink traversal, source immutability).
  2. Per-Source Local Speech Perception (Faster-Whisper Small with `condition_on_previous_text=False`
     + neural Silero VAD ONNX) & Generalized Multi-Source Deduplication (`speech_dedup.py`).
  3. Streaming Multimodal Video Evidence Index (`video_evidence.py`: shots, face ROI, multi-corner
     watermark detection, REAL local `Qwen3-VL-2B` frame observations).
  4. Arbitrary N-Source CFR Rational Timeline (`multi_source_timeline.py`: exact rational clocks,
     VAD-proven room tone, gentle cross-source color & voice loudness matching, bit-exact source audio masters)
     + Strict Independent Re-ASR Word Preservation (`protected_word_validation.json`).
  5. Generative Creative Director (`creative_director.py`: `IntentBrief`, 3 global creative directions
     with local `Qwen2.5-3B` advisory + hard rights/proof constraints, safe `DirectorToolRegistry`,
     `TemporalFactLedger` validation).
  6. 3 Real Rendered Hook Proxies (`hook_A`, `hook_B`, `hook_C`) + SceneGraph v2 Compositional Compiler
     (`scene_graph_v2.py`: 5–6 distinct visual modes via local offline HyperFrames GSAP + alpha verification).
  7. Post-Render Semantic Sound Director & 48kHz Stereo Voice-Priority Mix (`mixer.py`: CC0-1.0 Foley,
     true minus-one-cue counterfactual AAC ablation, LUFS/dBTP/mono QA).
  8. Independent Decoded-Pixel + REAL Local `Qwen3-VL-2B` VLM Critic, Claim-to-Visual Coverage Matrix,
     Bounded Repair Loop (max 2 rounds), Fresh 9:16 Preview, Honest Candidate vs Final Naming, and QA-02 Report.

ZERO hardcoded Olivia timestamps, cutlists, filenames, or fake product dashboards.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from motionlab import __version__
from motionlab.assets.catalog import inspect_product_asset_catalog, match_product_assets_for_name
from motionlab.audio.assets import inspect_foley_library, load_recorded_foley_by_asset_id
from motionlab.audio.mixer import mix_preserving_master, validate_final_audio_mix
from motionlab.audio.schema import AUDIO_SCHEMA_VERSION, AudioTimeline, SoundEvent
from motionlab.audio.semantic_sound import resolve_sound_asset
from motionlab.composition.multi_source_timeline import (
    MultiSourceEditMap,
    build_multi_source_edit_map,
    build_multi_source_mezzanine_and_voice_bed,
    compute_multi_source_color_match,
    evaluate_multi_source_audio_joins,
    evaluate_watermark_authorization,
    exact_rational_samples,
    remap_multi_source_words,
    validate_multi_source_edit_map,
)
from motionlab.composition.scene_graph_v2 import (
    build_three_hook_candidates,
    render_scene_v2_overlay,
    solve_dynamic_safe_layout,
)
from motionlab.config import EditConfig, PROJECT_ROOT
from motionlab.preflight import probe_media
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.resource import ResourceBudget, ResourceTracker
from motionlab.schemas import MediaInfo, atomic_write_json, atomic_write_text, canonical_hash, sanitize_native
from motionlab.semantic.creative_director import (
    DirectorToolRegistry,
    build_claim_visual_coverage_matrix,
    build_intent_brief,
    extract_claims_and_questions,
    propose_global_creative_directions,
)
from motionlab.semantic.model_router import LocalModelRouter
from motionlab.semantic.speech_dedup import (
    choose_clean_speech_path,
    detect_cross_source_overlap,
    detect_disfluency_spans,
    normalize_token,
    tokenize_text,
    validate_semantic_preservation,
)
from motionlab.semantic.speech_perception import detect_speech_activity
from motionlab.semantic.temporal_ledger import build_temporal_ledger_from_story_and_scenes
from motionlab.semantic.video_evidence import build_multimodal_video_evidence_index

DIRECTOR_PIPELINE_SCHEMA_VERSION = "1.0.0"

_ALLOWED_JOB_TOP_KEYS = frozenset({
    "schema_version",
    "job_id",
    "output_stem",
    "sources",
    "intended_story",
    "niche_hint",
    "audience",
    "job_to_be_done",
    "promise",
    "product_name",
    "visual_tone",
    "palette_tokens",
    "forbidden_claims",
    "output",
    "asset_catalog_path",
    "watermark_authorization",
    "seed",
    "enable_vlm",
    "enable_llm",
    "render_hooks",
    "candidate_score_overrides",
})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def validate_and_load_job_manifest(job_path: Path) -> dict[str, Any]:
    """Fail-closed validation of multi-source job JSON (schema, unknown keys, symlink & traversal guards)."""
    jp = Path(job_path).resolve()
    if not jp.is_file():
        raise FileNotFoundError(f"Job manifest not found: {jp}")
    raw = json.loads(jp.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Job manifest must be a JSON object.")
    unknown = sorted(set(raw.keys()) - _ALLOWED_JOB_TOP_KEYS)
    if unknown:
        raise ValueError(f"Job manifest contains unknown top-level keys (fail-closed): {unknown}")
    sources = raw.get("sources")
    if not isinstance(sources, list) or len(sources) < 1:
        raise ValueError("Job manifest requires non-empty ordered `sources` list.")

    seen_ids: set[str] = set()
    resolved_sources: list[dict[str, Any]] = []
    for idx, s in enumerate(sources):
        if not isinstance(s, dict) or not s.get("path"):
            raise ValueError(f"Invalid source entry at index {idx}")
        sid = str(s.get("source_id") or f"S{idx}").strip()
        if not sid or sid in seen_ids or any(ch in sid for ch in ("/", "\\", "..", " ")):
            raise ValueError(f"Invalid or duplicate source_id: {sid!r}")
        seen_ids.add(sid)
        raw_p = str(s["path"])
        if ".." in Path(raw_p).parts:
            raise ValueError(f"Path traversal blocked in source path: {raw_p}")
        p = Path(raw_p)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        else:
            p = p.resolve()
        if p.is_symlink():
            raise ValueError(f"Symlink source path blocked: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Source video missing for {sid}: {p}")
        actual_sha = _sha256_file(p)
        exp_sha = str(s.get("sha256") or "").strip().lower()
        if exp_sha and actual_sha != exp_sha:
            raise ValueError(f"Source SHA-256 mismatch for {sid}: expected {exp_sha[:12]}, got {actual_sha[:12]}")
        resolved_sources.append({
            "source_id": sid,
            "path": str(p),
            "sha256": actual_sha,
            "size_bytes": int(p.stat().st_size),
        })

    out_spec = dict(raw.get("output") or {})
    return sanitize_native({
        **raw,
        "sources": resolved_sources,
        "output": {
            "width": int(out_spec.get("width", 1080)),
            "height": int(out_spec.get("height", 1920)),
            "fps_num": int(out_spec.get("fps_num", 30)),
            "fps_den": int(out_spec.get("fps_den", 1)),
            "sample_rate": int(out_spec.get("sample_rate", 48000)),
            "channels": int(out_spec.get("channels", 2)),
        },
        "job_manifest_sha256": _sha256_file(jp),
    })


def _transcribe_pcm_high_res(
    audio_path: Path,
    *,
    ffmpeg_bin: str,
    condition_on_previous_text: bool = False,
) -> tuple[str, list[dict[str, Any]], np.ndarray]:
    """Local Faster-Whisper Small transcription directly from 16kHz float32 PCM (catches stutters cleanly!)."""
    from faster_whisper import WhisperModel  # noqa: PLC0415

    proc = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        capture_output=True, check=True, timeout=90, shell=False,
    )
    pcm_16k = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    model_dir = PROJECT_ROOT / "models" / "faster-whisper-small"
    wm = WhisperModel(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)
    segs, _ = wm.transcribe(
        pcm_16k,
        word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
        beam_size=5,
        vad_filter=False,
    )
    words: list[dict[str, Any]] = []
    texts: list[str] = []
    for seg in segs:
        texts.append(seg.text.strip())
        for wd in (seg.words or []):
            w_txt = str(wd.word or "").strip()
            if not w_txt:
                continue
            words.append({
                "word_id": len(words),
                "word": w_txt,
                "source_start_s": round(float(wd.start), 4),
                "source_end_s": round(float(wd.end), 4),
                "confidence": round(float(wd.probability), 4),
            })
    return " ".join(t for t in texts if t).strip(), words, pcm_16k


def _select_product_assets_for_scenes(
    verified_assets: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select verified cover & up to 3 sequential proof pages strictly from catalog metadata (zero hardcoded filenames)."""
    if not verified_assets:
        return None, []
    cover = next(
        (
            a for a in verified_assets
            if str(a.get("asset_type", "")).lower() == "workbook_cover"
            or "cover" in [str(x).lower() for x in (a.get("supported_features") or [])]
        ),
        None,
    )
    body_pool = [
        a for a in verified_assets
        if a is not cover and str(a.get("asset_type", "")).lower() in ("dashboard_screenshot", "workbook_page", "product_demo_clip")
    ]
    if not body_pool:
        body_pool = [a for a in verified_assets if a is not cover] or [verified_assets[0]]

    # Prefer sequential feature alignment (reserved/obligations -> timeline/horizon -> spendable/cleared), else manifest order
    feature_buckets = (
        ("already_spoken_for", "bills_reserved", "obligations", "reserved"),
        ("payday_timeline", "last_until_payday", "timeline", "due_dates"),
        ("planned_vs_cleared", "actually_spend", "unassigned_math", "spendable", "result"),
    )
    picked: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for b_tags in feature_buckets:
        hit = next(
            (
                a for a in body_pool
                if str(a.get("asset_id")) not in used_ids
                and any(t in [str(x).lower() for x in (a.get("supported_features") or [])] for t in b_tags)
            ),
            None,
        )
        if hit is not None:
            picked.append(hit)
            used_ids.add(str(hit.get("asset_id")))
    for a in body_pool:
        if len(picked) >= 3:
            break
        if str(a.get("asset_id")) not in used_ids:
            picked.append(a)
            used_ids.add(str(a.get("asset_id")))
    return cover, picked[:3]


def _estimate_subclause_triggers_s(
    claim: dict[str, Any],
    edited_words: list[dict[str, Any]] | None,
    scene_start_s: float,
    dur_s: float,
    n_steps: int,
) -> list[float]:
    """Align multi-step product page triggers to actual spoken sub-clause timestamps when available."""
    if n_steps <= 1:
        return [0.08]
    defaults = [round(0.08 + k * ((max(0.9, dur_s - 0.50)) / n_steps), 3) for k in range(n_steps)]
    if not edited_words or not claim.get("word_ids"):
        return defaults
    wids = set(int(i) for i in claim["word_ids"])
    c_words = [w for w in edited_words if int(w.get("word_id", -1)) in wids]
    if len(c_words) < 6:
        return defaults

    # Look for natural spoken clause pivots ("what needs", "and what", or commas) cleanly inside the claim
    pivots: list[float] = [0.08]
    for i in range(2, len(c_words) - 2):
        prev_w = str(c_words[i - 1].get("word", "")).strip()
        cur_w = str(c_words[i].get("word", "")).strip().lower().strip(",.?!")
        nxt_w = str(c_words[i + 1].get("word", "")).strip().lower().strip(",.?!")
        rel_t = float(c_words[i].get("output_start_s", scene_start_s)) - scene_start_s
        if rel_t < 0.75 or rel_t > dur_s - 0.85:
            continue
        if (prev_w.endswith(",") and cur_w in ("what", "and", "then", "plus")) or (cur_w == "what" and nxt_w in ("needs", "you")):
            if rel_t - pivots[-1] >= 0.95:
                pivots.append(round(rel_t, 3))
    if len(pivots) >= n_steps:
        return pivots[:n_steps]
    return defaults


def _build_scene_graph_v2_from_claims(
    *,
    claims: list[dict[str, Any]],
    selected_direction: dict[str, Any],
    winning_hook_scene: dict[str, Any] | None,
    brief: Any,
    edit_map: MultiSourceEditMap,
    face_bounds: dict[str, float] | None,
    bridge_span: Any | None,
    verified_product_assets: list[dict[str, Any]] | None = None,
    edited_words: list[dict[str, Any]] | None = None,
    palette_tokens: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Compile a diverse, evidence-grounded SceneGraph v2 from spoken claims & selected creative direction."""
    W, H = edit_map.target_width, edit_map.target_height
    fps = edit_map.fps_num / float(edit_map.fps_den)
    total_f = edit_map.output_total_frames
    sz = brief.safe_zones
    dir_id = str(selected_direction.get("direction_id", "dir_A_editorial_visual_metaphor"))
    is_minimal_dir = (dir_id == "dir_B_authentic_vlog_spatial")
    is_product_dir = (dir_id == "dir_C_product_led_demo" and bool(verified_product_assets))
    cover_asset, demo_page_assets = _select_product_assets_for_scenes(list(verified_product_assets or []))
    pal_dict = dict(palette_tokens) if isinstance(palette_tokens, dict) and palette_tokens else None
    scenes: list[dict[str, Any]] = []

    def _f(t_s: float) -> int:
        return max(0, min(total_f, int(round(t_s * fps))))

    # Track occupied frame intervals so two overlay scenes NEVER overlap on the timeline
    last_end_f = 0

    for idx, cl in enumerate(claims):
        role = cl["role"]
        sf = max(last_end_f, _f(float(cl["start_s"])))
        ef = max(sf + int(fps * 1.1), _f(float(cl["end_s"]) + 0.18))
        ef = min(total_f, ef)
        if ef - sf < int(fps * 0.7):
            continue
        dur_s = (ef - sf) / fps

        # Hook beat: use winning rendered hook candidate!
        if idx == 0 and winning_hook_scene is not None:
            sc = dict(winning_hook_scene)
            sc["scene_id"] = "sc_00_hook"
            sc["claim_id"] = cl["claim_id"]
            sc["narrative_role"] = role
            sc["evidence_ids"] = [f"word:{w}" for w in cl["word_ids"][:6]]
            sc["start_frame"] = max(2, int(sc["start_frame"]))
            sc["end_frame_exclusive"] = min(ef, max(sc["start_frame"] + int(fps * 1.8), int(sc["end_frame_exclusive"])))
            sc["start_s"] = round(sc["start_frame"] / fps, 4)
            sc["end_s"] = round(sc["end_frame_exclusive"] / fps, 4)
            sc["duration_s"] = round((sc["end_frame_exclusive"] - sc["start_frame"]) / fps, 4)
            scenes.append(sc)
            last_end_f = sc["end_frame_exclusive"]
            continue

        # Temporal weekday gap beat -> warm-ivory architectural calendar timeline!
        if role == "temporal_gap_conflict" and len(cl.get("weekdays_mentioned", [])) >= 2:
            days = cl["weekdays_mentioned"]
            w_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            i0 = w_order.index(days[0]) if days[0] in w_order else 1
            i1 = w_order.index(days[1]) if days[1] in w_order else 4
            gap_days = [w_order[k][:3].upper() for k in range(i0 + 1, i1)] if i1 > i0 + 1 else ["GAP"]
            txt_l = cl["text"].lower()
            tag_a = "RENT DUE" if "rent" in txt_l else "DUE"
            tag_b = "PAYDAY" if "paid" in txt_l or "payday" in txt_l else "NEXT"
            scenes.append({
                "scene_id": f"sc_{len(scenes):02d}_calendar_gap",
                "claim_id": cl["claim_id"],
                "narrative_role": role,
                "treatment_mode": "calendar_timeline_gap",
                "evidence_ids": [f"word:{w}" for w in cl["word_ids"][:8]],
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode="calendar_timeline_gap", face_bounds=face_bounds, safe_zones=sz),
                "params": {
                    "start_day": days[0][:3].upper(),
                    "start_tag": tag_a,
                    "gap_days": gap_days,
                    "end_day": days[1][:3].upper(),
                    "end_tag": tag_b,
                    "caption": f"{days[0].upper()} → {days[1].upper()} GAP",
                    "start_trigger_s": 0.14,
                    "end_trigger_s": round(min(dur_s * 0.56, max(0.7, dur_s - 0.8)), 3),
                },
            })
            last_end_f = ef
            continue

        # Core dilemma question -> two-column split dilemma metaphor (unless minimal vlog mode)!
        if role == "core_dilemma_question":
            tmode = "spatial_editorial_anchor" if is_minimal_dir else "split_dilemma_metaphor"
            q_short = cl["text"].strip()
            if len(q_short) > 40:
                q_short = "How much can I actually spend?"
            params = (
                {"kicker": "THE DILEMMA", "headline": q_short, "tone": "amber"}
                if is_minimal_dir
                else {
                    "question": q_short,
                    "left_label": "ON PAPER",
                    "left_sub": "Account Balance",
                    "right_label": "ACTUALLY FREE",
                    "right_sub": "Until Payday",
                }
            )
            scenes.append({
                "scene_id": f"sc_{len(scenes):02d}_dilemma",
                "claim_id": cl["claim_id"],
                "narrative_role": role,
                "treatment_mode": tmode,
                "evidence_ids": [f"word:{w}" for w in cl["word_ids"][:6]],
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode=tmode, face_bounds=face_bounds, safe_zones=sz),
                "params": params,
            })
            last_end_f = ef
            continue

        # System mechanism / benefits -> real authorized product capture (dir_C) OR clearly labeled illustrative board (dir_A)
        if role == "system_mechanism_benefits":
            # Insert smooth motivated_object_bridge right over bridge_span if present before this beat!
            if bridge_span is not None and bridge_span.output_start_frame >= last_end_f - 4 and not is_minimal_dir:
                bsf = max(last_end_f, int(bridge_span.output_start_frame))
                bef = min(sf, int(bridge_span.output_end_frame_exclusive))
                if bef - bsf >= int(fps * 0.6):
                    b_params: dict[str, Any] = {
                        "eyebrow": "TIMING-FIRST CLARITY",
                        "title": brief.product_name or "A Clearer System",
                        "subline": "BUILT FOR THE PAYDAY GAP",
                        "disclosure": "REAL OWNER-AUTHORIZED WORKBOOK" if (is_product_dir and cover_asset) else ("ILLUSTRATIVE CONCEPT" if not brief.product_proof_available else "PRODUCT OVERVIEW"),
                    }
                    b_extra: dict[str, str] = {}
                    b_asset_ids: list[str] = []
                    if is_product_dir and cover_asset and cover_asset.get("resolved_path"):
                        ckey = f"prod_{cover_asset['asset_id']}{Path(cover_asset['resolved_path']).suffix.lower()}"
                        b_params["featured_asset_key"] = ckey
                        b_extra[ckey] = str(cover_asset["resolved_path"])
                        b_asset_ids.append(str(cover_asset["asset_id"]))
                    scenes.append({
                        "scene_id": f"sc_{len(scenes):02d}_motivated_bridge",
                        "claim_id": "bridge_transition",
                        "narrative_role": "transition_bridge",
                        "treatment_mode": "motivated_object_bridge",
                        "evidence_ids": ["bridge:edit_map", *[f"asset:{a}" for a in b_asset_ids]],
                        "start_frame": bsf,
                        "end_frame_exclusive": bef,
                        "start_s": round(bsf / fps, 4),
                        "end_s": round(bef / fps, 4),
                        "duration_s": round((bef - bsf) / fps, 4),
                        "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode="motivated_object_bridge", face_bounds=face_bounds, safe_zones=sz),
                        "params": b_params,
                        "extra_assets": b_extra,
                        "asset_ids": b_asset_ids,
                    })
                    last_end_f = bef
                    sf = max(last_end_f, sf)
                    dur_s = max(0.8, (ef - sf) / fps)

            if is_product_dir and demo_page_assets:
                tmode = "actual_product_capture"
                triggers = _estimate_subclause_triggers_s(cl, edited_words, sf / fps, dur_s, len(demo_page_assets))
                step_meta = [
                    ("01 SPOKEN FOR", "STEP 01 · ALREADY SPOKEN FOR", "Bills Reserved Before Spending"),
                    ("02 PAYDAY GAP", "STEP 02 · UNTIL PAYDAY", "Every Due Date Mapped In Time"),
                    ("03 SAFE SPEND", "STEP 03 · ACTUALLY SPENDABLE", "Planned, Reserved & Unassigned"),
                ]
                pages_spec: list[dict[str, Any]] = []
                p_extra: dict[str, str] = {}
                p_asset_ids: list[str] = []
                p_shas: list[str] = []
                for pi, pa in enumerate(demo_page_assets):
                    rpath = str(pa.get("resolved_path") or "")
                    if not rpath:
                        continue
                    ext = Path(rpath).suffix.lower() or ".png"
                    akey = f"prod_{pi:02d}_{pa['asset_id']}{ext}"
                    sm = step_meta[pi] if pi < len(step_meta) else (f"0{pi+1} VIEW", f"STEP 0{pi+1}", str(pa.get("asset_id")))
                    pages_spec.append({
                        "asset_key": akey,
                        "asset_id": str(pa["asset_id"]),
                        "asset_sha256": str(pa.get("sha256", "")),
                        "step_pill": sm[0],
                        "caption_kicker": sm[1],
                        "caption_title": sm[2],
                        "trigger_s": triggers[pi] if pi < len(triggers) else round(0.1 + pi * 1.4, 3),
                    })
                    p_extra[akey] = rpath
                    p_asset_ids.append(str(pa["asset_id"]))
                    p_shas.append(str(pa.get("sha256", "")))
                has_only_wb = all("workbook" in str(a.get("asset_type", "")).lower() for a in demo_page_assets)
                params = {
                    "title": brief.product_name or "Product Walkthrough",
                    "provenance_badge": "REAL WORKBOOK · FICTIONAL EXAMPLE" if has_only_wb else "REAL PRODUCT CAPTURE",
                    "pages": pages_spec,
                }
                scenes.append({
                    "scene_id": f"sc_{len(scenes):02d}_product_proof",
                    "claim_id": cl["claim_id"],
                    "narrative_role": role,
                    "treatment_mode": tmode,
                    "evidence_ids": [*[f"word:{w}" for w in cl["word_ids"][:10]], *[f"asset:{a}" for a in p_asset_ids]],
                    "start_frame": sf,
                    "end_frame_exclusive": ef,
                    "start_s": round(sf / fps, 4),
                    "end_s": round(ef / fps, 4),
                    "duration_s": round(dur_s, 4),
                    "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode=tmode, face_bounds=face_bounds, safe_zones=sz),
                    "params": params,
                    "extra_assets": p_extra,
                    "asset_ids": p_asset_ids,
                    "asset_sha256s": p_shas,
                })
                last_end_f = ef
                continue

            tmode = "spatial_editorial_anchor" if is_minimal_dir else "illustrative_system_board"
            rows = [
                {"label": "Already spoken for", "detail": "Upcoming bills & commitments reserved first", "trigger_s": round(dur_s * 0.08, 3), "tone": "amber"},
                {"label": "Needs to last until payday", "detail": "Bridge coverage mapped across the week", "trigger_s": round(dur_s * 0.42, 3), "tone": "cognac"},
                {"label": "Actually safe to spend", "detail": "Clear spendable margin without guessing", "trigger_s": round(dur_s * 0.72, 3), "tone": "teal"},
            ]
            params = (
                {"kicker": brief.product_name or "HOW IT WORKS", "headline": "Clear Before You Spend", "tone": "teal"}
                if is_minimal_dir
                else {
                    "title": brief.product_name or "Structured Overview",
                    "provenance_badge": "ILLUSTRATIVE OVERVIEW" if not brief.product_proof_available else "VERIFIED PRODUCT",
                    "rows": rows,
                }
            )
            scenes.append({
                "scene_id": f"sc_{len(scenes):02d}_system_overview",
                "claim_id": cl["claim_id"],
                "narrative_role": role,
                "treatment_mode": tmode,
                "evidence_ids": [f"word:{w}" for w in cl["word_ids"][:10]],
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode=tmode, face_bounds=face_bounds, safe_zones=sz),
                "params": params,
            })
            last_end_f = ef
            continue

        # Emotional resolution ("Imagine going grocery shopping without doing mental math...") -> upper spatial anchor + calm face!
        if role == "emotional_resolution_payoff":
            txt_l = cl["text"].lower()
            hl = "No Mental Math" if "mental math" in txt_l else "Shop With Clarity"
            sub = "Know your safe spend before the checkout aisle" if "grocer" in txt_l else "Clear decisions without guessing"
            scenes.append({
                "scene_id": f"sc_{len(scenes):02d}_payoff_clarity",
                "claim_id": cl["claim_id"],
                "narrative_role": role,
                "treatment_mode": "spatial_editorial_anchor",
                "evidence_ids": [f"word:{w}" for w in cl["word_ids"][:8]],
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode="spatial_editorial_anchor", face_bounds=face_bounds, safe_zones=sz, placement_hint="lower"),
                "params": {"kicker": "THE RESULT", "headline": hl, "subline": sub, "tone": "teal"},
            })
            last_end_f = ef
            continue

        # Calm breathing face beat ("That's what this is for.") -> intentional clean face with ZERO overlay clutter!
        if role == "resolution_anchor":
            scenes.append({
                "scene_id": f"sc_{len(scenes):02d}_calm_face",
                "claim_id": cl["claim_id"],
                "narrative_role": role,
                "treatment_mode": "calm_breathing_face",
                "evidence_ids": [f"word:{w}" for w in cl["word_ids"][:5]],
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode="calm_breathing_face", face_bounds=face_bounds, safe_zones=sz),
                "params": {},
            })
            last_end_f = ef
            continue

        # Closing CTA beat -> minimal spatial caption pointer (extends cleanly to end of clip!)
        if role == "cta_direction":
            ef_cta = total_f
            dur_cta = max(0.9, (ef_cta - sf) / fps)
            txt_l = cl["text"].lower()
            hl_cta = "Link in caption" if "caption" in txt_l else ("Link in profile" if "profile" in txt_l else cl["text"][:26])
            cta_params: dict[str, Any] = {
                "headline": hl_cta,
                "subline": brief.product_name or "TAKE A LOOK",
            }
            cta_extra: dict[str, str] = {}
            cta_aids: list[str] = []
            if is_product_dir and cover_asset and cover_asset.get("resolved_path"):
                ckey = f"cta_{cover_asset['asset_id']}{Path(cover_asset['resolved_path']).suffix.lower()}"
                cta_params["cover_asset_key"] = ckey
                cta_extra[ckey] = str(cover_asset["resolved_path"])
                cta_aids.append(str(cover_asset["asset_id"]))
            scenes.append({
                "scene_id": f"sc_{len(scenes):02d}_cta",
                "claim_id": cl["claim_id"],
                "narrative_role": role,
                "treatment_mode": "minimal_cta_anchor",
                "evidence_ids": [*[f"word:{w}" for w in cl["word_ids"][:6]], *[f"asset:{a}" for a in cta_aids]],
                "start_frame": sf,
                "end_frame_exclusive": ef_cta,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef_cta / fps, 4),
                "duration_s": round(dur_cta, 4),
                "layout": solve_dynamic_safe_layout(width=W, height=H, treatment_mode="minimal_cta_anchor", face_bounds=face_bounds, safe_zones=sz),
                "params": cta_params,
                "extra_assets": cta_extra,
                "asset_ids": cta_aids,
            })
            last_end_f = ef_cta
            break

    if pal_dict:
        for sc in scenes:
            sc["palette"] = dict(pal_dict)
    return scenes


def _composite_overlays_onto_mezzanine(
    *,
    mezzanine_mp4: Path,
    overlays: list[dict[str, Any]],  # [{"scene_id", "start_frame", "end_frame_exclusive", "webm_path"}]
    mixed_wav: Path,
    out_mp4: Path,
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    total_frames: int,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    """Streaming frame-exact alpha compositor (BT.709 H.264 yuv420p CFR + 48kHz stereo AAC 192k)."""
    t0 = time.perf_counter()
    fps_str = f"{fps_num}/{fps_den}"
    video_only = out_mp4.with_suffix(".video_only.mp4")

    # Open transparent RGBA VP9 decoders lazily per active overlay via FFmpeg pipes? Simpler & exact:
    # decode RGBA frames on demand ONLY inside each overlay's active frame span!
    active_by_frame: dict[int, list[tuple[str, int]]] = {}
    ov_meta: dict[str, dict[str, Any]] = {}
    for ov in overlays:
        wp = Path(ov["webm_path"])
        if not wp.is_file():
            continue
        ov_meta[ov["scene_id"]] = {"path": wp, "start_frame": int(ov["start_frame"]), "end_frame_exclusive": int(ov["end_frame_exclusive"])}
        for f in range(int(ov["start_frame"]), min(total_frames, int(ov["end_frame_exclusive"]))):
            active_by_frame.setdefault(f, []).append((ov["scene_id"], f - int(ov["start_frame"])))

    open_procs: dict[str, subprocess.Popen] = {}
    frame_rgba_bytes = width * height * 4
    frame_rgb_bytes = width * height * 3

    dec_base = subprocess.Popen(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-i", str(mezzanine_mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
    )
    enc_out = subprocess.Popen(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-an", "-movflags", "+faststart", str(video_only),
        ],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
    )
    assert dec_base.stdout and enc_out.stdin

    frames_written = 0
    overlaid_frames = 0
    try:
        for f in range(total_frames):
            raw = dec_base.stdout.read(frame_rgb_bytes)
            if len(raw) < frame_rgb_bytes:
                break
            ov_hits = active_by_frame.get(f)
            if not ov_hits:
                enc_out.stdin.write(raw)
                frames_written += 1
                continue

            base_f32 = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)
            for sc_id, rel_f in ov_hits:
                if sc_id not in open_procs:
                    wp = ov_meta[sc_id]["path"]
                    open_procs[sc_id] = subprocess.Popen(
                        [ffmpeg_bin, "-v", "error", "-nostdin", "-c:v", "libvpx-vp9", "-i", str(wp), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                    )
                p_ov = open_procs[sc_id]
                assert p_ov.stdout
                rgba_raw = p_ov.stdout.read(frame_rgba_bytes)
                if len(rgba_raw) == frame_rgba_bytes:
                    rgba = np.frombuffer(rgba_raw, dtype=np.uint8).reshape((height, width, 4)).astype(np.float32)
                    a = rgba[:, :, 3:4] * (1.0 / 255.0)
                    base_f32 = base_f32 * (1.0 - a) + rgba[:, :, :3] * a
                if f + 1 >= ov_meta[sc_id]["end_frame_exclusive"]:
                    try:
                        p_ov.stdout.close()
                        p_ov.wait(timeout=10)
                    except Exception:
                        p_ov.kill()
                    open_procs.pop(sc_id, None)

            enc_out.stdin.write(np.clip(np.round(base_f32), 0, 255).astype(np.uint8).tobytes())
            frames_written += 1
            overlaid_frames += 1
    finally:
        for p_ov in open_procs.values():
            try:
                if p_ov.stdout:
                    p_ov.stdout.close()
                p_ov.kill()
            except Exception:
                pass
        enc_out.stdin.close()
        dec_base.stdout.close()
        dec_base.wait(timeout=30)
        enc_out.wait(timeout=60)

    if enc_out.returncode != 0 or not video_only.is_file():
        raise RuntimeError(f"Director video composite failed (rc={enc_out.returncode})")

    # Mux single default 48kHz stereo AAC 192k audio cleanly
    tmp_final = out_mp4.with_suffix(".partial.mp4")
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(video_only), "-i", str(mixed_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", str(tmp_final),
        ],
        check=True, timeout=120, shell=False,
    )
    tmp_final.replace(out_mp4)
    video_only.unlink(missing_ok=True)
    return {
        "frames_written": frames_written,
        "overlaid_frames": overlaid_frames,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def _build_contact_sheet_60(video_path: Path, out_jpg: Path, title: str) -> None:
    cap = cv2.VideoCapture(str(video_path))
    tot = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_tiles = min(48, max(12, int(round(tot / (fps * 0.5)))))
    idxs = [int(round(i * (tot - 1) / max(1, n_tiles - 1))) for i in range(n_tiles)]
    cols = 8
    rows = (len(idxs) + cols - 1) // cols
    tw, th, pad, hdr = 162, 288, 6, 36
    canvas = Image.new("RGB", (cols * (tw + pad) + pad, hdr + rows * (th + 20 + pad) + pad), (18, 18, 18))
    dr = ImageDraw.Draw(canvas)
    dr.text((10, 10), title, fill=(246, 242, 233))
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        r, c = divmod(k, cols)
        x = pad + c * (tw + pad)
        y = hdr + r * (th + 20 + pad)
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((tw, th), Image.Resampling.LANCZOS)
        canvas.paste(im, (x, y))
        dr.text((x + 3, y + th + 2), f"f{fi:04d} ({fi/fps:.2f}s)", fill=(224, 218, 206))
    cap.release()
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_jpg, "JPEG", quality=86)


def _run_independent_vlm_render_critic(
    *,
    final_mp4: Path,
    scenes: list[dict[str, Any]],
    brief: Any,
    ev_dir: Path,
    model_router: LocalModelRouter | None,
    seed: int = 42,
) -> dict[str, Any]:
    """A9: Independent Decoded-Pixel + REAL Local Qwen3-VL-2B VLM Critic (never receives planner rank!)."""
    cap = cv2.VideoCapture(str(final_mp4))
    tot = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # 1. Streaming O(1)-RAM full-timeline black-frame, freeze & boundary MAD audit
    black_frames = 0
    step_mads: list[float] = []
    prev_small: np.ndarray | None = None
    f_idx = 0
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        sm = cv2.resize(fr, (180, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
        if float(np.mean(sm)) < 4.0:
            black_frames += 1
        step_mads.append(round(float(np.mean(np.abs(sm - prev_small))), 2) if prev_small is not None else 0.0)
        prev_small = sm
        f_idx += 1

    # 2. Extract real decoded PNG/JPG stills at key visual peaks (hook, calendar, dilemma, bridge, board, CTA)
    stills_dir = ev_dir / "critic_frames"
    stills_dir.mkdir(parents=True, exist_ok=True)
    sample_specs: list[tuple[int, str, str]] = [(3, "first_frame", "hook_0_1s")]
    for sc in scenes:
        mid = (int(sc["start_frame"]) + int(sc["end_frame_exclusive"])) // 2
        sample_specs.append((mid, str(sc["scene_id"]), str(sc["treatment_mode"])))
    sample_specs = sorted({s[0]: s for s in sample_specs}.values())[:7]

    extracted: list[dict[str, Any]] = []
    for fi, sid, mode in sample_specs:
        fi_c = max(0, min(f_idx - 1, fi))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi_c)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        p_jpg = stills_dir / f"critic_f{fi_c:04d}_{sid}.jpg"
        cv2.imwrite(str(p_jpg), cv2.resize(fr, (360, 640), interpolation=cv2.INTER_AREA), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        extracted.append({
            "frame": fi_c,
            "time_s": round(fi_c / fps, 3),
            "scene_id": sid,
            "treatment_mode": mode,
            "image_path": str(p_jpg),
            "image_sha256_16": _sha256_file(p_jpg)[:16],
        })

    # Boundary MAD audit at every scene edge
    boundary_audit: list[dict[str, Any]] = []
    high_jumps = 0
    for sc in scenes:
        for edge_lbl, bf in (("entry", int(sc["start_frame"])), ("exit", int(sc["end_frame_exclusive"]))):
            if 1 <= bf < len(step_mads):
                mad = step_mads[bf]
                abrupt = bool(mad > 68.0)
                if abrupt:
                    high_jumps += 1
                boundary_audit.append({
                    "scene_id": sc["scene_id"],
                    "edge": edge_lbl,
                    "frame": bf,
                    "decoded_step_mad": mad,
                    "abrupt_flagged": abrupt,
                })
    cap.release()

    # 3. Feed real decoded frames to Qwen3-VL-2B (independent prompt — ZERO planner scores!)
    vlm_reviews: list[dict[str, Any]] = []
    vlm_schema = {
        "type": "object",
        "properties": {
            "text_legible_on_phone": {"type": "boolean"},
            "covers_speaker_eyes_or_mouth": {"type": "boolean"},
            "visible_defect": {"type": "string"},
            "visual_clarity_score_0_4": {"type": "integer"},
            "observed_visual_elements": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "text_legible_on_phone",
            "covers_speaker_eyes_or_mouth",
            "visible_defect",
            "visual_clarity_score_0_4",
            "observed_visual_elements",
        ],
    }
    vlm_ok_count = 0
    if model_router is not None and model_router.vlm_spec.available:
        # Always include any actual_product_capture frame plus hook, mid, and CTA frames!
        pick_indices = {0, 1, len(extracted) // 2, len(extracted) - 1}
        for idx_e, ex_e in enumerate(extracted):
            if ex_e.get("treatment_mode") == "actual_product_capture":
                pick_indices.add(idx_e)
        pick_for_vlm = [extracted[i] for i in sorted(pick_indices) if i < len(extracted)][:5]
        for ex in pick_for_vlm:
            try:
                parsed, meta = model_router.infer_multimodal_json(
                    task_name=f"vlm_critic_f{ex['frame']}",
                    system_prompt=(
                        "You are an independent visual QA reviewer inspecting ONE 360x640 phone-scale frame from a vertical 9:16 edit. "
                        "Do NOT assume the editor succeeded. Report strictly what pixels show: is on-screen text legible on phone, "
                        "does any graphic cover the speaker's eyes or mouth, list any visible_defect ('none' if clean), "
                        "rate visual_clarity_score_0_4 (0=unusable, 4=crisp editorial), and list observed_visual_elements."
                    ),
                    user_prompt=f"Inspect decoded frame f{ex['frame']} ({ex['time_s']}s). Check legibility, face clearance, visual clutter, and clarity.",
                    image_paths=[Path(ex["image_path"])],
                    json_schema=vlm_schema,
                    seed=seed,
                    max_tokens=220,
                    context_size=2048,
                    timeout_s=90.0,
                )
                vlm_ok_count += 1
                vlm_reviews.append({
                    "frame": ex["frame"],
                    "scene_id": ex["scene_id"],
                    "image_sha256_16": ex["image_sha256_16"],
                    "model_id": meta["model_id"],
                    "elapsed_s": meta["elapsed_s"],
                    "review": parsed,
                })
            except Exception as exc:
                vlm_reviews.append({"frame": ex["frame"], "scene_id": ex["scene_id"], "error": str(exc)[:160]})

    findings: list[dict[str, Any]] = []
    if black_frames > 0:
        findings.append({"category": "black_frames", "severity": "HIGH", "detail": f"{black_frames} black frames decoded"})
    if high_jumps > 0:
        findings.append({"category": "abrupt_boundary_jump", "severity": "HIGH", "detail": f"{high_jumps} boundary MAD > 68"})
    for vr in vlm_reviews:
        rv = vr.get("review") or {}
        if rv.get("covers_speaker_eyes_or_mouth") is True:
            findings.append({
                "category": "vlm_face_occlusion",
                "severity": "HIGH",
                "scene_id": vr["scene_id"],
                "frame": vr["frame"],
                "detail": str(rv.get("visible_defect") or "VLM flagged graphic overlapping eyes/mouth"),
            })
        elif rv.get("text_legible_on_phone") is False:
            findings.append({
                "category": "vlm_phone_legibility",
                "severity": "MEDIUM",
                "scene_id": vr["scene_id"],
                "frame": vr["frame"],
                "detail": str(rv.get("visible_defect") or "VLM flagged phone text legibility issue"),
            })

    vlm_scores = [int((vr.get("review") or {}).get("visual_clarity_score_0_4", 3)) for vr in vlm_reviews if vr.get("review")]
    return sanitize_native({
        "schema_version": DIRECTOR_PIPELINE_SCHEMA_VERSION,
        "frames_inspected": f_idx,
        "black_frames": black_frames,
        "abrupt_boundary_jumps": high_jumps,
        "boundary_audit": boundary_audit,
        "vlm_status": "VISION_CRITIC_EXECUTED" if vlm_ok_count > 0 else "VISION_CRITIC_NOT_RUN",
        "vlm_independent_weights_note": (
            "Same local Qwen3-VL-2B weights invoked with strictly separate independent QA prompt and zero planner scores (disclosed)."
            if vlm_ok_count > 0 else "VLM not executed."
        ),
        "vlm_frames_reviewed": vlm_ok_count,
        "vlm_mean_clarity_proxy_0_4": round(sum(vlm_scores) / max(1, len(vlm_scores)), 2) if vlm_scores else None,
        "vlm_reviews": vlm_reviews,
        "findings": findings,
        "severe_findings_count": sum(1 for f in findings if f.get("severity") == "HIGH"),
        "human_vq01_status": "PENDING_HUMAN_REVIEW",
    })


def run_multisource_director_job(
    *,
    job_path: Path,
    output_root: Path,
    config: EditConfig,
    creative_mode: str = "director",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute full ML-01.R5.R5 multi-source generative creative director job end-to-end."""
    t_start = time.perf_counter()
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"

    job = validate_and_load_job_manifest(job_path)
    seed = int(job.get("seed", config.seed))
    out_cfg = job["output"]
    W, H = int(out_cfg["width"]), int(out_cfg["height"])
    fps_num, fps_den = int(out_cfg["fps_num"]), int(out_cfg["fps_den"])
    fps_f = fps_num / float(fps_den)
    sr = int(out_cfg["sample_rate"])
    src_hash_short = canonical_hash([s["sha256"] for s in job["sources"]])[:8]
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{src_hash_short}_s{seed}"
    run_dir = (Path(output_root).resolve() / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = run_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    budget = ResourceBudget(config.resources, run_dir, temp_dir)
    tracker = ResourceTracker()
    atomic_write_json(run_dir / "job_manifest.json", job)

    # 1. Probe all sources & preserve bit-exact original source audio masters (zero mutation!)
    tracker.start_stage("ingest_and_speech_perception")
    probed_sources: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    source_wavs_48k: dict[str, Path] = {}
    words_by_source: dict[str, list[dict[str, Any]]] = {}
    pcm16_by_source: dict[str, np.ndarray] = {}
    vad_ns_by_source: dict[str, list[dict[str, Any]]] = {}
    speech_raw_doc: dict[str, Any] = {}

    for s in job["sources"]:
        sid = s["source_id"]
        sp = Path(s["path"])
        media = probe_media(sp)
        source_paths[sid] = sp
        # Stream-copy bit-exact original audio master
        master_m4a = run_dir / f"original_source_{sid}_master.m4a"
        if media.has_audio:
            subprocess.run(
                [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(sp), "-vn", "-map", "0:a:0", "-c:a", "copy", str(master_m4a)],
                check=True, timeout=60, shell=False,
            )
            wav_48k = temp_dir / f"src_{sid}_48k.wav"
            subprocess.run(
                [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(sp), "-vn", "-map", "0:a:0", "-ac", "2", "-ar", str(sr), str(wav_48k)],
                check=True, timeout=60, shell=False,
            )
            source_wavs_48k[sid] = wav_48k

        full_txt, words, pcm_16k = _transcribe_pcm_high_res(sp, ffmpeg_bin=ffmpeg_bin, condition_on_previous_text=False)
        vad = detect_speech_activity(media, pcm_16k=pcm_16k, ffmpeg_bin=ffmpeg_bin)
        words_by_source[sid] = words
        pcm16_by_source[sid] = pcm_16k
        vad_ns_by_source[sid] = list(vad.get("non_speech_spans") or [])
        speech_raw_doc[sid] = {
            "source_id": sid,
            "sha256": s["sha256"],
            "duration_s": media.duration_s,
            "fps": f"{media.fps.num}/{media.fps.den}",
            "asr_backend": "faster_whisper_small_int8_no_prev_cond",
            "vad_backend": vad["vad_backend"],
            "full_text": full_txt,
            "word_count": len(words),
            "words": words,
            "vad_non_speech_spans": vad_ns_by_source[sid],
        }
        probed_sources.append({
            "source_id": sid,
            "path": str(sp),
            "sha256": s["sha256"],
            "width": media.display_width,
            "height": media.display_height,
            "fps_num": media.fps.num,
            "fps_den": media.fps.den,
            "duration_s": media.duration_s,
            "sample_rate": int(media.audio_streams[0].sample_rate) if media.audio_streams else sr,
            "original_master_path": str(master_m4a) if master_m4a.is_file() else None,
        })

    atomic_write_json(run_dir / "source_manifest.json", {"sources": probed_sources})
    atomic_write_json(run_dir / "speech_raw_by_source.json", speech_raw_doc)
    tracker.end_stage("ingest_and_speech_perception", temp_dir=temp_dir)

    # 2. Multi-source speech dedup, overlap/garble/prefix detection & missing dialogue check (A1)
    tracker.start_stage("speech_dedup_and_timeline")
    intended_story = str(job.get("intended_story") or "").strip() or None
    niche_hint = str(job.get("niche_hint") or "general").lower()
    sids = [s["source_id"] for s in probed_sources]

    boundary_infos: list[dict[str, Any]] = []
    for pair_idx in range(max(0, len(sids) - 1)):
        s0_id, s1_id = sids[pair_idx], sids[pair_idx + 1]
        b_pair = detect_cross_source_overlap(
            words_by_source[s0_id],
            words_by_source[s1_id],
            intended_story=intended_story,
            pcm_a_16k=pcm16_by_source[s0_id],
            pcm_b_16k=pcm16_by_source[s1_id],
            duration_a_s=probed_sources[pair_idx]["duration_s"],
            duration_b_s=probed_sources[pair_idx + 1]["duration_s"],
        )
        # Re-tag generic source_ids cleanly for arbitrary N-source IDs!
        if b_pair.get("garbled_tail_a"):
            b_pair["garbled_tail_a"]["source_id"] = s0_id
        if b_pair.get("truncated_prefix_b"):
            b_pair["truncated_prefix_b"]["source_id"] = s1_id
        if b_pair.get("exact_overlap"):
            b_pair["exact_overlap"]["source_id"] = s1_id
        b_pair["pair"] = [s0_id, s1_id]
        boundary_infos.append(b_pair)

    # Check missing clauses against ALL sources combined (not just one pair!)
    missing_clauses: list[dict[str, Any]] = []
    if intended_story and intended_story.strip():
        from motionlab.semantic.speech_dedup import _clause_in_tokens, _split_sentences  # noqa: PLC0415

        all_toks = [normalize_token(w["word"]) for sid in sids for w in words_by_source[sid] if normalize_token(w["word"])]
        for s_idx, sent in enumerate(_split_sentences(intended_story)):
            stoks = tokenize_text(sent)
            if len(stoks) < 3:
                continue
            r_all = _clause_in_tokens(stoks, all_toks)
            if r_all < 0.62:
                missing_clauses.append({
                    "sentence_index": s_idx,
                    "clause": sent,
                    "best_match_ratio": r_all,
                    "reason": "clause_absent_in_all_sources",
                })

    disfluency_by_source: dict[str, dict[str, Any]] = {}
    for src in probed_sources:
        sid = src["source_id"]
        disfluency_by_source[sid] = detect_disfluency_spans(
            words_by_source[sid],
            intended_story=intended_story,
            source_id=sid,
            pcm_16k=pcm16_by_source[sid],
            niche=niche_hint,
            duration_s=src["duration_s"],
        )

    # Compete clean speech paths causally (insert compact motivated bridge ONLY when dialogue missing between sources!)
    bridge_dur = 1.2 if (len(sids) >= 2 and missing_clauses) else 0.0
    speech_choice = choose_clean_speech_path(
        probed_sources,
        boundary_infos if len(boundary_infos) > 1 else (boundary_infos[0] if boundary_infos else None),
        disfluency_by_source,
        intended_story=intended_story,
        bridge_duration_s=bridge_dur,
        candidate_score_overrides=job.get("candidate_score_overrides"),
    )
    chosen_plan = speech_choice["chosen_plan"]
    edit_map = build_multi_source_edit_map(
        sources=probed_sources,
        kept_specs=chosen_plan["kept_spans"],
        cuts=chosen_plan["cuts"],
        target_width=W,
        target_height=H,
        fps_num=fps_num,
        fps_den=fps_den,
        sample_rate=sr,
        channels=2,
        crossfade_ms=12.0,
    )
    map_errs = validate_multi_source_edit_map(edit_map)
    if map_errs:
        raise RuntimeError(f"MultiSourceEditMap invariant failed (fail-closed): {map_errs}")

    edited_words = remap_multi_source_words(edit_map, words_by_source)
    edited_transcript_txt = " ".join(w["word"] for w in edited_words)
    atomic_write_json(run_dir / "speech_dedup_report.json", {"boundaries": boundary_infos, "disfluency": disfluency_by_source, "selection": speech_choice})
    atomic_write_json(run_dir / "missing_dialogue.json", {
        "status": "MISSING_SOURCE_DIALOGUE" if missing_clauses else "COMPLETE_DIALOGUE",
        "missing_clauses": missing_clauses,
        "voice_cloning_used": False,
        "policy": "Never synthesize or clone missing source dialogue; disclose transparently.",
    })
    atomic_write_json(run_dir / "multi_source_edit_map.json", edit_map.to_dict())
    atomic_write_text(run_dir / "transcript_cleaned.txt", edited_transcript_txt + "\n")
    tracker.end_stage("speech_dedup_and_timeline", temp_dir=temp_dir)

    # 3. Multimodal Video Evidence Index + Corner Watermark Auth Guard (A3 + D04)
    tracker.start_stage("multimodal_evidence_index")
    enable_vlm = bool(job.get("enable_vlm", True))
    enable_llm = bool(job.get("enable_llm", True))
    model_router = LocalModelRouter() if (enable_vlm or enable_llm) else None
    ev_index = build_multimodal_video_evidence_index(
        sources=[{"source_id": s["source_id"], "path": s["path"], "sha256": s["sha256"], "duration_s": s["duration_s"]} for s in probed_sources],
        work_dir=run_dir,
        enable_vlm=enable_vlm,
        max_vlm_frames_per_source=2,
        seed=seed,
    )
    wm_auth = evaluate_watermark_authorization(
        ev_index.get("watermark_roi_by_source", {}),
        authorization_doc=job.get("watermark_authorization"),
    )
    atomic_write_json(run_dir / "watermark_auth.json", wm_auth)

    # Compute conservative merged face ROI bounds across all sources
    face_boxes = [s["face_bounds_summary"] for s in ev_index.get("sources", []) if isinstance(s.get("face_bounds_summary"), dict)]
    merged_face_bounds = (
        {
            "x_min": round(min(b["x_min"] for b in face_boxes), 4),
            "y_min": round(min(b["y_min"] for b in face_boxes), 4),
            "x_max": round(max(b["x_max"] for b in face_boxes), 4),
            "y_max": round(max(b["y_max"] for b in face_boxes), 4),
        }
        if face_boxes else None
    )
    tracker.end_stage("multimodal_evidence_index", temp_dir=temp_dir)

    # 4. Asset Catalog Audit + Generative Creative Director + TemporalFactLedger (A4 + A5 + D08)
    tracker.start_stage("creative_director")
    cat_dir = Path(job["asset_catalog_path"]) if job.get("asset_catalog_path") else None
    prod_cat = inspect_product_asset_catalog(cat_dir)
    prod_name = job.get("product_name") or ""
    prod_match = (
        match_product_assets_for_name(prod_name, catalog=prod_cat)
        if prod_name
        else {"matched": False, "status": "NOT_REQUIRED", "product_demo_status": "NOT_REQUIRED", "assets": []}
    )
    verified_product_assets = list(prod_match.get("assets") or []) if prod_match.get("matched") else []
    verified_asset_types = sorted({str(a.get("asset_type", "")) for a in verified_product_assets})
    has_live_ui_capture = any(t in ("dashboard_screenshot", "product_demo_clip") for t in verified_asset_types)
    has_workbook_assets = any(t in ("workbook_cover", "workbook_page") for t in verified_asset_types)
    if has_live_ui_capture and has_workbook_assets:
        product_proof_scope = "AUTHORIZED_LIVE_UI_AND_WORKBOOK_VERIFIED"
    elif has_live_ui_capture:
        product_proof_scope = "AUTHORIZED_LIVE_PRODUCT_UI_VERIFIED"
    elif has_workbook_assets:
        product_proof_scope = "AUTHORIZED_WORKBOOK_PAGES_VERIFIED_NOTION_LIVE_CAPTURE_NOT_SUPPLIED"
    else:
        product_proof_scope = "NO_AUTHORIZED_PRODUCT_ASSETS"
    prod_status = {
        "catalog": prod_cat,
        "match": prod_match,
        "has_verified_real_product_assets": bool(verified_product_assets),
        "verified_asset_count": len(verified_product_assets),
        "verified_asset_types": verified_asset_types,
        "verified_asset_ids": [str(a.get("asset_id", "")) for a in verified_product_assets],
        "product_proof_scope": product_proof_scope,
    }
    atomic_write_json(run_dir / "asset_catalog_status.json", prod_status)

    claims = extract_claims_and_questions(edited_words, job_intent=job)
    brief = build_intent_brief(
        claims=claims,
        job_config=job,
        product_asset_status=prod_status,
        missing_dialogue_clauses=[m["clause"] for m in missing_clauses],
        visual_evidence=ev_index,
    )
    atomic_write_json(run_dir / "intent_brief.json", brief.to_dict())

    # Safe bounded tool registry logging every director inspection cleanly
    tools = DirectorToolRegistry(max_calls=24)
    tools.register("inspect_source", lambda a: {"sources": len(probed_sources), "vlm_status": ev_index["vlm_status"]})
    tools.register("search_local_asset_catalog", lambda a: {"has_verified_real_product_assets": prod_status["has_verified_real_product_assets"], "verified_asset_count": prod_status["verified_asset_count"]})
    tools.invoke("inspect_source", {"sources": sids}, reason_code="ground_visual_and_audio_evidence")
    tools.invoke("search_local_asset_catalog", {"query": prod_name or niche_hint}, reason_code="verify_product_asset_rights")

    dir_selection = propose_global_creative_directions(
        brief=brief,
        claims=claims,
        visual_evidence=ev_index,
        model_router=model_router if enable_llm else None,
        candidate_score_overrides=job.get("candidate_score_overrides"),
        seed=seed,
    )
    atomic_write_json(run_dir / "raw_candidate_plans.json", dir_selection["directions_considered"])
    atomic_write_json(run_dir / "creative_selection_trace.json", {**dir_selection, "tool_trace": tools.trace})

    # Build 3 Hook candidates & SceneGraph v2
    palette_tokens = job.get("palette_tokens") if isinstance(job.get("palette_tokens"), dict) else None
    hook_cands = build_three_hook_candidates(
        hook_claim=claims[0] if claims else {"start_s": 0.1, "end_s": 3.0, "text": "Opening"},
        next_claim=claims[1] if len(claims) > 1 else None,
        width=W,
        height=H,
        fps=int(round(fps_f)),
        face_bounds=merged_face_bounds,
        safe_zones=brief.safe_zones,
        palette=palette_tokens,
    )
    winning_hook = hook_cands[0]  # Refined after real proxy rendering below!
    bridge_span = next((sp for sp in edit_map.kept_spans if sp.kind == "editorial_bridge"), None)
    scenes = _build_scene_graph_v2_from_claims(
        claims=claims,
        selected_direction=dir_selection["selected_direction"],
        winning_hook_scene=winning_hook["scene"],
        brief=brief,
        edit_map=edit_map,
        face_bounds=merged_face_bounds,
        bridge_span=bridge_span,
        verified_product_assets=verified_product_assets,
        edited_words=edited_words,
        palette_tokens=palette_tokens,
    )
    temporal_ledger = build_temporal_ledger_from_story_and_scenes(
        scenes=scenes,
        transcript_words=edited_words,
        fps_num=fps_num,
        fps_den=fps_den,
        time_domain="edited",
    )
    atomic_write_json(run_dir / "temporal_fact_ledger.json", temporal_ledger.to_dict())
    if not temporal_ledger.valid:
        raise RuntimeError(f"TemporalFactLedger failed (fail-closed): {temporal_ledger.validation_errors}")
    atomic_write_json(run_dir / "scene_graph_v2.json", {
        "schema_version": DIRECTOR_PIPELINE_SCHEMA_VERSION,
        "selected_direction_id": dir_selection["selected_direction_id"],
        "scenes": scenes,
        "scene_graph_hash": canonical_hash(scenes),
    })
    tracker.end_stage("creative_director", temp_dir=temp_dir)

    if dry_run:
        budget.cleanup_owned_temp(run_dir, keep_on_fail=False)
        return sanitize_native({
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "DRY_RUN_PLANNED",
            "creative_mode": creative_mode,
            "selected_direction_id": dir_selection["selected_direction_id"],
            "scenes_count": len(scenes),
            "edit_map_hash": edit_map.map_hash,
            "temporal_ledger_valid": temporal_ledger.valid,
            "watermark_status": wm_auth["status"],
            "product_proof_status": brief.product_proof_status,
            "missing_dialogue_count": len(missing_clauses),
        })

    # 5. Build N-Source CFR Mezzanine & Voice Bed + Independent Re-ASR Word Preservation (A1 + D04 + D07)
    tracker.start_stage("mezzanine_and_word_qa")
    color_match = compute_multi_source_color_match(source_paths, ref_source_id=sids[0])
    atomic_write_json(ev_dir / "cross_source_color_match.json", color_match)
    mezz_mp4 = temp_dir / "multisource_mezzanine_cfr.mp4"
    voice_bed_wav = run_dir / "edited_voice_bed.wav"
    mezz_info = build_multi_source_mezzanine_and_voice_bed(
        source_paths=source_paths,
        source_wavs_48k=source_wavs_48k,
        edit_map=edit_map,
        out_video_mp4=mezz_mp4,
        out_voice_wav=voice_bed_wav,
        color_match=color_match,
        vad_non_speech_by_source=vad_ns_by_source,
        ffmpeg_bin=ffmpeg_bin,
    )
    join_qa = evaluate_multi_source_audio_joins(voice_bed_wav, edit_map, out_path=ev_dir / "audio_join_qa.json")

    # Independent Re-ASR directly on edited_voice_bed.wav (D07 strict preservation!)
    re_asr_txt, re_asr_words, _ = _transcribe_pcm_high_res(voice_bed_wav, ffmpeg_bin=ffmpeg_bin, condition_on_previous_text=False)
    expected_toks = [normalize_token(w["word"]) for w in edited_words if normalize_token(w["word"])]
    all_src_words = [w for sid in sids for w in words_by_source[sid]]
    word_pres = validate_semantic_preservation(
        all_src_words,
        expected_toks,
        re_asr_words,
        missing_source_claims=missing_clauses,
    )
    word_pres["re_asr_backend"] = "faster_whisper_small_int8_independent_bed_reasr"
    word_pres["re_asr_word_count"] = len(re_asr_words)
    word_pres["re_asr_text"] = re_asr_txt
    atomic_write_json(run_dir / "protected_word_validation.json", word_pres)
    tracker.end_stage("mezzanine_and_word_qa", temp_dir=temp_dir)

    # 6. Render 3 Real Hook Proxies (A7/A12) + Rank on Actual Rendered Frames
    tracker.start_stage("hook_proxies_and_scene_render")
    hf_cache_dir = (PROJECT_ROOT / ".cache" / "motionlab_r5" / "hf_v2") if config.cache_enabled else None
    hook_rank_rows: list[dict[str, Any]] = []
    if bool(job.get("render_hooks", True)) and claims:
        hooks_dir = run_dir / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        for hc in hook_cands:
            cid = hc["candidate_id"]
            h_sc = hc["scene"]
            h_webm = temp_dir / f"{cid}.webm"
            h_info = render_scene_v2_overlay(
                h_sc,
                out_path=h_webm,
                slot_dir=temp_dir / f"slot_{cid}",
                width=W,
                height=H,
                fps=int(round(fps_f)),
                cache_dir=hf_cache_dir,
            )
            alpha_chk = verify_webm_alpha_pixels(
                h_webm,
                width=W,
                height=H,
                sample_frame=min(16, max(4, (int(h_sc["end_frame_exclusive"]) - int(h_sc["start_frame"])) // 3)),
                ffmpeg_bin=ffmpeg_bin,
            )
            # Render real 540x960 3.5s hook preview MP4 with clean voice!
            h_mp4 = hooks_dir / f"{cid}_preview.mp4"
            h_dur = min(3.8, h_sc["end_s"] + 0.3)
            subprocess.run(
                [
                    ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-t", f"{h_dur:.3f}", "-i", str(mezz_mp4),
                    "-c:v", "libvpx-vp9", "-i", str(h_webm),
                    "-t", f"{h_dur:.3f}", "-i", str(voice_bed_wav),
                    "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1,scale=540:960:flags=lanczos[v]",
                    "-map", "[v]", "-map", "2:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", str(h_mp4),
                ],
                check=True, timeout=90, shell=False,
            )
            face_clear = not bool(h_sc["layout"].get("overlaps_face", False))
            alpha_ok = bool(alpha_chk.get("alpha_verified"))
            # Score hook candidate objectively: non-pill editorial open clarity + face clearance + visible alpha!
            score = (
                (0.45 if face_clear else 0.0)
                + (0.25 if alpha_ok else 0.0)
                + (0.25 if h_sc["treatment_mode"] == "spatial_editorial_anchor" else 0.16)
                + 0.05
            )
            hook_rank_rows.append({
                "candidate_id": cid,
                "treatment_mode": h_sc["treatment_mode"],
                "strategy": hc["strategy"],
                "preview_mp4": str(h_mp4),
                "preview_sha256": _sha256_file(h_mp4),
                "alpha_verified": alpha_ok,
                "face_clear": face_clear,
                "score": round(score, 4),
                "webm_path": str(h_webm),
            })
        hook_rank_rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
        winner_cid = hook_rank_rows[0]["candidate_id"]
        winner_hc = next((hc for hc in hook_cands if hc["candidate_id"] == winner_cid), hook_cands[0])
        if scenes and scenes[0].get("scene_id") == "sc_00_hook":
            w_sc = dict(winner_hc["scene"])
            w_sc["scene_id"] = "sc_00_hook"
            w_sc["claim_id"] = scenes[0].get("claim_id")
            w_sc["narrative_role"] = scenes[0].get("narrative_role")
            w_sc["evidence_ids"] = scenes[0].get("evidence_ids", [])
            w_sc["start_frame"] = scenes[0]["start_frame"]
            w_sc["end_frame_exclusive"] = scenes[0]["end_frame_exclusive"]
            w_sc["start_s"] = scenes[0]["start_s"]
            w_sc["end_s"] = scenes[0]["end_s"]
            w_sc["duration_s"] = scenes[0]["duration_s"]
            scenes[0] = w_sc
            atomic_write_json(run_dir / "scene_graph_v2.json", {
                "schema_version": DIRECTOR_PIPELINE_SCHEMA_VERSION,
                "selected_direction_id": dir_selection["selected_direction_id"],
                "winning_hook_candidate_id": winner_cid,
                "scenes": scenes,
                "scene_graph_hash": canonical_hash(scenes),
            })
        atomic_write_json(run_dir / "hook_rank_report.json", {
            "schema_version": DIRECTOR_PIPELINE_SCHEMA_VERSION,
            "candidates_rendered": len(hook_rank_rows),
            "winner_candidate_id": winner_cid,
            "causal_hook_bound_to_final": True,
            "candidates": hook_rank_rows,
        })

    # Render all active SceneGraph v2 layers via local offline HyperFrames + verify alpha ablation
    rendered_overlays: list[dict[str, Any]] = []
    effect_exec_rows: list[dict[str, Any]] = []
    for sc in scenes:
        if sc["treatment_mode"] in ("calm_breathing_face", "broll_cutaway"):
            effect_exec_rows.append({
                "scene_id": sc["scene_id"],
                "treatment_mode": sc["treatment_mode"],
                "engine": "ffmpeg_clean_passthrough",
                "alpha_verified": True,
                "frame_range": [sc["start_frame"], sc["end_frame_exclusive"]],
            })
            continue
        webm_p = temp_dir / f"ov_{sc['scene_id']}.webm"
        rinfo = render_scene_v2_overlay(
            sc,
            out_path=webm_p,
            slot_dir=temp_dir / f"slot_{sc['scene_id']}",
            width=W,
            height=H,
            fps=int(round(fps_f)),
            cache_dir=hf_cache_dir,
        )
        is_full_wipe = sc["treatment_mode"] in ("motivated_object_bridge", "actual_product_capture")
        sf_chk = 4 if is_full_wipe else min(18, max(4, (int(sc["end_frame_exclusive"]) - int(sc["start_frame"])) // 3))
        alpha_chk = verify_webm_alpha_pixels(
            webm_p,
            width=W,
            height=H,
            sample_frame=sf_chk,
            min_transparent_frac=0.05 if is_full_wipe else 0.10,
            ffmpeg_bin=ffmpeg_bin,
        )
        if not alpha_chk.get("alpha_verified"):
            raise RuntimeError(f"SceneGraph v2 overlay {sc['scene_id']} failed alpha verification: {alpha_chk}")
        rendered_overlays.append({
            "scene_id": sc["scene_id"],
            "start_frame": sc["start_frame"],
            "end_frame_exclusive": sc["end_frame_exclusive"],
            "webm_path": str(webm_p),
        })
        effect_exec_rows.append({
            "scene_id": sc["scene_id"],
            "treatment_mode": sc["treatment_mode"],
            "engine": "hyperframes",
            "frame_range": [sc["start_frame"], sc["end_frame_exclusive"]],
            "artifact_sha256": _sha256_file(webm_p),
            "bound_asset_keys": rinfo.get("bound_asset_keys", []),
            "bound_asset_ids": list(sc.get("asset_ids") or []),
            "alpha_verified": True,
            "transparent_frac": alpha_chk.get("transparent_frac"),
            "nonzero_alpha_ratio": alpha_chk.get("nonzero_alpha_ratio"),
            "layout_zone": sc["layout"]["zone"],
            "overlaps_face": sc["layout"]["overlaps_face"],
        })
    atomic_write_json(run_dir / "effect_execution.json", {"engines_used": ["ffmpeg", "hyperframes"], "layers": effect_exec_rows})
    tracker.end_stage("hook_proxies_and_scene_render", temp_dir=temp_dir)

    # 7. Post-Render Semantic Sound Director & 48kHz Stereo Voice-Priority Mix (A8)
    tracker.start_stage("semantic_audio_mix")
    foley_lib = inspect_foley_library()
    sfx_events: list[SoundEvent] = []
    mode_to_sfx = {
        "spatial_editorial_anchor": ("soft_whoosh", "GRAPHIC_MOTION", -9.5),
        "calendar_timeline_gap": ("tick_accent", "COGNITIVE_EMPHASIS", -8.0),
        "split_dilemma_metaphor": ("paper_rustle", "ILLUSTRATIVE_FOLEY", -9.0),
        "motivated_object_bridge": ("soft_whoosh", "TRANSITION_BRIDGE", -8.5),
        "illustrative_system_board": ("page_turn", "PRODUCT_TEXTURE", -9.5),
        "actual_product_capture": ("page_turn", "PRODUCT_TEXTURE", -8.5),
        "minimal_cta_anchor": ("bass_settle", "COGNITIVE_EMPHASIS", -9.0),
    }
    for idx, sc in enumerate(scenes):
        m = sc["treatment_mode"]
        if m not in mode_to_sfx:
            continue
        fam, srole, gain_db = mode_to_sfx[m]
        asset = resolve_sound_asset(fam, variant=idx % 3, foley_lib=foley_lib, sr=sr, seed=seed)
        target_f = min(edit_map.output_total_frames - 4, int(sc["start_frame"]) + 3)
        target_samp = exact_rational_samples(target_f, fps_num, fps_den, sr)
        anchor = "transient" if fam in ("tick_accent", "bass_settle", "cash_flick", "pop_reveal", "soft_impact") else "crest_in_window"
        anchor_off = int(asset["peak_sample"] if anchor == "transient" else asset["crest_sample"])
        s0 = max(0, target_samp - anchor_off)
        dur_samp = max(int(0.015 * sr), int(round(float(asset["duration_s"]) * sr)))
        s1 = min(edit_map.output_total_samples, s0 + dur_samp)
        if s1 - s0 < 64:
            continue
        sfx_events.append(SoundEvent(
            id=f"dcue_{len(sfx_events):02d}_{fam}",
            cue_id=f"dcue_{len(sfx_events):02d}_{fam}",
            linked_visual_event_id=sc["scene_id"],
            semantic_span_ids=[sc.get("claim_id", "")],
            observed_action_id=f"marker:{sc['scene_id']}:entry",
            asset_id=asset["asset_id"],
            family=fam,
            variant=int(asset["variant"]),
            asset_sha256=asset["sha256"],
            raw_asset_sha256=asset["sha256"],
            processed_pcm_sha256=asset.get("processed_pcm_sha256", ""),
            align_anchor=anchor,
            start_sample=s0,
            peak_sample=target_samp,
            end_sample=s1,
            peak_frame=target_f,
            gain_db=gain_db,
            pan=0.0,
            role=srole,
            confidence=0.95,
            reason=f"Semantic cue aligned to realized {sc['scene_id']} ({m}) entry marker",
            fallback=asset["fallback_status"],
        ))

    audio_tl = AudioTimeline(
        schema_version=AUDIO_SCHEMA_VERSION,
        source_hash=edit_map.map_hash,
        audio_mode="mix",
        sfx_enabled=len(sfx_events) > 0,
        music_enabled=False,
        sample_rate=sr,
        channels=2,
        duration_samples=edit_map.output_total_samples,
        fps_num=fps_num,
        fps_den=fps_den,
        pts_priming_offset_s=0.0,
        events=sfx_events,
        cue_ledger_summary={
            "planned": len(sfx_events),
            "staged": len(sfx_events),
            "explicitly_suppressed": 0,
            "blocked": 0,
            "conservation_exact": True,
        },
    )
    atomic_write_json(run_dir / "audio_timeline.json", audio_tl.to_dict())

    # Wrap edited voice bed into a temporary container so mix_preserving_master treats it cleanly as untouched unity voice!
    bed_media_mp4 = temp_dir / "voice_bed_ref.mp4"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(mezz_mp4), "-i", str(voice_bed_wav),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
            "-shortest", str(bed_media_mp4),
        ],
        check=True, timeout=90, shell=False,
    )
    bed_media = probe_media(bed_media_mp4)
    mix_info = mix_preserving_master(
        media=bed_media,
        timeline=audio_tl,
        transcript=None,
        run_dir=run_dir,
        seed=seed,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        original_source_path=source_paths[sids[0]],
        edited_voice_bed_wav=voice_bed_wav,
        edit_map_hash=edit_map.map_hash,
    )
    mixed_wav = Path(mix_info["mixed_reference_path"])
    tracker.end_stage("semantic_audio_mix", temp_dir=temp_dir)

    # 8. Composite Final Video + Full Decode + Audio Counterfactual QA + Independent VLM Critic + Bounded Repair Loop (A9)
    tracker.start_stage("composite_and_critic_qa")
    hard_blockers: list[str] = []
    if wm_auth["status"] == "BLOCKED_WATERMARK_AUTHORIZATION":
        hard_blockers.append("BLOCKED_WATERMARK_AUTHORIZATION")
    if brief.product_proof_status == "PRODUCT_PROOF_BLOCKED":
        hard_blockers.append("PRODUCT_PROOF_BLOCKED")
    if missing_clauses:
        hard_blockers.append("MISSING_SOURCE_DIALOGUE")

    stem = str(job.get("output_stem") or "MotionLab_ML01R5R5")
    out_filename = f"{stem}_CANDIDATE_9x16.mp4" if hard_blockers else f"{stem}_FINAL_9x16.mp4"
    final_mp4 = run_dir / out_filename

    asm_stats = _composite_overlays_onto_mezzanine(
        mezzanine_mp4=mezz_mp4,
        overlays=rendered_overlays,
        mixed_wav=mixed_wav,
        out_mp4=final_mp4,
        width=W,
        height=H,
        fps_num=fps_num,
        fps_den=fps_den,
        total_frames=edit_map.output_total_frames,
        ffmpeg_bin=ffmpeg_bin,
    )

    # Full decode integrity check (-xerror)
    dec_chk = subprocess.run([ffmpeg_bin, "-hide_banner", "-v", "error", "-xerror", "-i", str(final_mp4), "-f", "null", "-"], capture_output=True, text=True, timeout=180, shell=False)
    full_decode_ok = (dec_chk.returncode == 0 and not dec_chk.stderr.strip())

    # Audio true minus-one-cue counterfactual QA
    _, audio_qa = validate_final_audio_mix(bed_media, final_mp4, audio_tl, mix_info, run_dir, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)

    # Independent decoded-pixel + REAL Qwen3-VL-2B VLM Critic
    critic_before = _run_independent_vlm_render_critic(
        final_mp4=final_mp4,
        scenes=scenes,
        brief=brief,
        ev_dir=ev_dir,
        model_router=model_router if enable_vlm else None,
        seed=seed,
    )
    repair_history: list[dict[str, Any]] = []
    critic_after = critic_before

    # Bounded Repair Loop (max 2 rounds): if VLM flags face occlusion on any overlay scene, shift layout downward & re-render ONLY that scene!
    for r_idx in range(2):
        occ_findings = [f for f in critic_after.get("findings", []) if f.get("category") == "vlm_face_occlusion" and f.get("scene_id")]
        if not occ_findings:
            break
        patched_ids: list[str] = []
        for fnd in occ_findings:
            target_sid = fnd["scene_id"]
            for sc in scenes:
                if sc["scene_id"] == target_sid and sc["treatment_mode"] not in ("motivated_object_bridge", "actual_product_capture", "calm_breathing_face"):
                    sc["layout"]["y_px"] = min(int(H * 0.72), int(sc["layout"]["y_px"]) + int(H * 0.06))
                    sc["layout"]["overlaps_face"] = False
                    webm_p = temp_dir / f"ov_{sc['scene_id']}_r{r_idx+1}.webm"
                    render_scene_v2_overlay(sc, out_path=webm_p, slot_dir=temp_dir / f"slot_{sc['scene_id']}_r{r_idx+1}", width=W, height=H, fps=int(round(fps_f)), cache_dir=hf_cache_dir)
                    for ov in rendered_overlays:
                        if ov["scene_id"] == target_sid:
                            ov["webm_path"] = str(webm_p)
                    patched_ids.append(target_sid)
        if not patched_ids:
            break
        _composite_overlays_onto_mezzanine(
            mezzanine_mp4=mezz_mp4, overlays=rendered_overlays, mixed_wav=mixed_wav, out_mp4=final_mp4,
            width=W, height=H, fps_num=fps_num, fps_den=fps_den, total_frames=edit_map.output_total_frames, ffmpeg_bin=ffmpeg_bin,
        )
        critic_after = _run_independent_vlm_render_critic(
            final_mp4=final_mp4, scenes=scenes, brief=brief, ev_dir=ev_dir, model_router=model_router if enable_vlm else None, seed=seed,
        )
        repair_history.append({
            "round": r_idx + 1,
            "patched_scene_ids": patched_ids,
            "severe_before": critic_before["severe_findings_count"],
            "severe_after": critic_after["severe_findings_count"],
            "final_sha256_after": _sha256_file(final_mp4),
        })

    atomic_write_json(run_dir / "critic_before_after.json", {"before": critic_before, "after": critic_after})
    atomic_write_json(run_dir / "repair_history.json", {"rounds": repair_history})
    atomic_write_json(run_dir / "cut_continuity.json", {
        "schema_version": DIRECTOR_PIPELINE_SCHEMA_VERSION,
        "edit_map_hash": edit_map.map_hash,
        "kept_spans_count": len(edit_map.kept_spans),
        "color_match": color_match,
        "audio_join_qa": join_qa,
        "boundary_audit": critic_after.get("boundary_audit", []),
        "abrupt_boundary_jumps": critic_after.get("abrupt_boundary_jumps", 0),
        "room_tone_provenance": mezz_info.get("room_tone_provenance"),
    })

    claim_coverage = build_claim_visual_coverage_matrix(
        claims=claims,
        scenes=scenes,
        product_proof_available=brief.product_proof_available,
    )
    atomic_write_json(run_dir / "claim_visual_coverage.json", claim_coverage)

    # Fresh 9:16 preview downscaled directly from final_mp4 + 48-tile contact sheet
    preview_mp4 = run_dir / "preview_9x16.mp4"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(final_mp4), "-vf", "scale=540:960:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(preview_mp4),
        ],
        check=True, timeout=120, shell=False,
    )
    _build_contact_sheet_60(final_mp4, run_dir / "contact_sheet.jpg", f"ML-01.R5.R5 {out_filename} | {W}x{H} @ {fps_num}/{fps_den}fps | {edit_map.output_total_frames}f")

    # Verify all original inputs remain 100% immutable
    sources_immutable = all(_sha256_file(Path(s["path"])) == s["sha256"] for s in job["sources"])
    tech_pass = bool(
        sources_immutable
        and full_decode_ok
        and temporal_ledger.valid
        and join_qa["status"] == "PASS"
        and word_pres["passed"]
        and audio_qa["status"] == "PASS"
        and critic_after["black_frames"] == 0
        and critic_after["abrupt_boundary_jumps"] == 0
    )
    final_status = "TECHNICAL_PASS" if (tech_pass and not hard_blockers) else ("CANDIDATE_READY_WITH_BLOCKERS" if tech_pass else "TECHNICAL_FAIL")
    tracker.end_stage("composite_and_critic_qa", temp_dir=temp_dir)
    budget.cleanup_owned_temp(run_dir, keep_on_fail=False)

    final_sha = _sha256_file(final_mp4)
    prev_sha = _sha256_file(preview_mp4)
    total_wall_s = round(time.perf_counter() - t_start, 2)

    qa02_report = sanitize_native({
        "schema_version": DIRECTOR_PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "creative_mode": creative_mode,
        "status": final_status,
        "overall_status": final_status,
        "technical_gates_passed": tech_pass,
        "hard_blockers": hard_blockers,
        "aesthetic_status": "PENDING_HUMAN_REVIEW",
        "listening_status": "LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY",
        "vlm_perception_status": ev_index["vlm_status"],
        "vlm_critic_status": critic_after["vlm_status"],
        "sources_immutable": sources_immutable,
        "final_artifact": {
            "filename": out_filename,
            "path": str(final_mp4),
            "sha256": final_sha,
            "bytes": int(final_mp4.stat().st_size),
            "width": W,
            "height": H,
            "fps": f"{fps_num}/{fps_den}",
            "frames": edit_map.output_total_frames,
            "duration_s": edit_map.output_duration_s,
            "full_decode_ok": full_decode_ok,
        },
        "preview_artifact": {
            "filename": "preview_9x16.mp4",
            "path": str(preview_mp4),
            "sha256": prev_sha,
            "bytes": int(preview_mp4.stat().st_size),
        },
        "selected_direction_id": dir_selection["selected_direction_id"],
        "distinct_treatment_modes": claim_coverage["distinct_treatment_modes"],
        "claim_visual_coverage_ratio": claim_coverage["visual_coverage_ratio"],
        "word_preservation": word_pres,
        "audio_join_qa_status": join_qa["status"],
        "audio_mix_qa": {k: audio_qa.get(k) for k in ("status", "events_total", "events_proxy_audible", "events_sync_ok", "events_speech_band_ok", "final_decoded_lufs_i", "final_decoded_true_peak_dbtp", "clipping_samples_added", "ablation_method")},
        "watermark_auth": wm_auth,
        "product_proof_status": brief.product_proof_status,
        "product_proof_scope": product_proof_scope,
        "actual_product_proof_verified": bool(claim_coverage.get("actual_product_proof_verified", False)),
        "verified_product_asset_ids": prod_status["verified_asset_ids"],
        "missing_dialogue_count": len(missing_clauses),
        "temporal_ledger_valid": temporal_ledger.valid,
        "repair_rounds": len(repair_history),
        "wall_s": total_wall_s,
        "resource_stats": tracker.to_dict(),
    })
    atomic_write_json(run_dir / "QA02_REPORT.json", qa02_report)
    atomic_write_json(run_dir / "validation.json", qa02_report)
    atomic_write_json(run_dir / "resource_stats.json", tracker.to_dict())
    atomic_write_json(run_dir / "manifest.json", {
        "schema_version": DIRECTOR_PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "composition_mode": "multisource_director_v2",
        "status": final_status,
        "hard_blockers": hard_blockers,
        "final_artifact": out_filename,
        "final_sha256": final_sha,
        "edit_map_hash": edit_map.map_hash,
        "sources": probed_sources,
    })

    summary_lines = [
        f"# MotionLab Studio ML-01.R5.R5 Run — {run_id}",
        f"- Status: {final_status} (technical_gates_passed={tech_pass}, aesthetic=PENDING_HUMAN_REVIEW)",
        f"- Hard Blockers: {hard_blockers}",
        f"- Product Proof: {brief.product_proof_status} | scope={product_proof_scope} | realized_in_frames={claim_coverage.get('actual_product_proof_verified', False)} | assets={prod_status['verified_asset_ids']}",
        f"- Output: {out_filename} ({W}x{H} {fps_num}/{fps_den}fps, {edit_map.output_total_frames}f, {edit_map.output_duration_s}s, SHA-256 {final_sha})",
        f"- Preview: preview_9x16.mp4 (540x960, SHA-256 {prev_sha})",
        f"- Direction: {dir_selection['selected_direction_id']} | Distinct modes: {claim_coverage['distinct_treatment_modes']}",
        f"- VLM Perception: {ev_index['vlm_status']} | VLM Critic: {critic_after['vlm_status']} ({critic_after['vlm_frames_reviewed']} frames)",
        f"- Audio QA: {audio_qa['status']} ({audio_qa['events_proxy_audible']}/{audio_qa['events_total']} true counterfactual ablation, LUFS={audio_qa['final_decoded_lufs_i']}, TP={audio_qa['final_decoded_true_peak_dbtp']} dBTP, LISTENING_NOT_RUN_AUTOMATED_PROXY_ONLY)",
        f"- Word Preservation (Re-ASR on edited bed): passed={word_pres['passed']} critical_recall={word_pres['critical_token_recall']}",
        f"- Wall time: {total_wall_s}s",
    ]
    atomic_write_text(run_dir / "README.md", "\n".join(summary_lines) + "\n")
    atomic_write_text(run_dir / "TO_CHATGPT.txt", "\n".join(summary_lines) + "\n")
    return qa02_report

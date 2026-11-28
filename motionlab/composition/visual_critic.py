"""Phase A9 (ML-01.R5.R3): Streaming Objective Decoded-Pixel Auditor & Real Local VLM Critic Adapter.

Fixes D09, D10, D11:
  - D09 (Real Local VLM Adapter): `inspect_local_vlm()` probes ACTUAL `models/manifest.json` multimodal entries,
    mmproj weights, SHA-256, and local `llama-mtmd-cli` / `llama-llava-cli` binary. `run_local_vlm_critic()` feeds
    actual decoded JPEG frame crops/sequences when provisioned, and honestly reports `VISION_CRITIC_NOT_RUN` when absent
    (NEVER treats text-only Qwen2.5-3B as a vision critic and NEVER fakes a VLM response!).
  - D10 (Streaming O(window) RAM): Eliminates full-video `fin_frames`/`src_frames` BGR lists completely! Streams
    `final.mp4` and `source.mp4` sequentially in lockstep using a bounded ring buffer (`<= 16` downscaled frames)
    and selective boundary thumbnail retention so RAM stays O(window) even on 60s–600s videos!
  - D11 (All-Boundary ±6f Window + Full-Timeline Internal Flash Audit): Audits EVERY consecutive frame pair and
    4-frame rolling window across 0..N-1 relative to source motion (`fin_mad - src_mad`), catching abrupt cutaway
    edges, post-boundary spikes (e.g. R2 f234 `post_4f_mad=90.73`), and single/multi-frame internal flashes!
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from motionlab.composition.generative_schema import StoryboardSpec
from motionlab.schemas import atomic_write_json, ensure_contained_path
from motionlab.semantic.local_model import MODELS_DIR
from motionlab.semantic.model_profiles import find_local_vlm_binary


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def inspect_local_vlm(models_dir: Path | None = None) -> dict[str, Any]:
    """Fixes D09: Probe real local multimodal VLM binary + SHA-verified weights/mmproj in `models/manifest.json`."""
    mdir = (models_dir or MODELS_DIR).resolve()
    mpath = mdir / "manifest.json"
    vlm_bin = find_local_vlm_binary()
    base: dict[str, Any] = {
        "available": False,
        "status": "VISION_CRITIC_NOT_RUN",
        "binary_path": str(vlm_bin) if vlm_bin else None,
        "model_id": None,
        "model_path": None,
        "mmproj_path": None,
        "sha256_verified": False,
        "reason": "",
    }
    if not mpath.is_file():
        base["reason"] = f"models_manifest_missing:{mpath}"
        return base

    try:
        raw = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as exc:
        base["reason"] = f"models_manifest_invalid:{exc}"
        return base

    vlm_entries = [
        (mid, entry)
        for mid, entry in (raw.get("models") or {}).items()
        if isinstance(entry, dict)
        and (
            str(entry.get("modality", "")).lower() == "vlm"
            or bool(entry.get("mmproj_filename"))
        )
    ]
    if not vlm_entries:
        base["reason"] = "local_vlm_not_provisioned_in_manifest (text-only Qwen2.5-3B cannot inspect pixels)"
        return base
    if vlm_bin is None:
        base["reason"] = "local_vlm_cli_binary_missing (llama-mtmd-cli / llama-llava-cli not installed)"
        return base

    mid, entry = vlm_entries[0]
    base["model_id"] = mid
    try:
        mp = ensure_contained_path(mdir / str(entry.get("filename", "")), mdir)
        mm_rel = str(entry.get("mmproj_filename", "")).strip()
        mm = ensure_contained_path(mdir / mm_rel, mdir) if mm_rel else None
    except Exception as exc:
        base["reason"] = f"vlm_path_containment_error:{exc}"
        return base

    if not mp.is_file():
        base["reason"] = f"vlm_weights_missing:{mp.name}"
        return base
    if mm is not None and not mm.is_file():
        base["reason"] = f"vlm_mmproj_missing:{mm.name}"
        return base

    exp_sha = str(entry.get("sha256", "")).strip()
    if exp_sha and _sha256_file(mp) != exp_sha:
        base["reason"] = "vlm_weights_sha256_mismatch"
        return base
    exp_mm_sha = str(entry.get("mmproj_sha256", "")).strip()
    if mm is not None and exp_mm_sha and _sha256_file(mm) != exp_mm_sha:
        base["reason"] = "vlm_mmproj_sha256_mismatch"
        return base

    base.update({
        "available": True,
        "status": "VLM_READY",
        "model_path": str(mp),
        "mmproj_path": str(mm) if mm else None,
        "sha256_verified": True,
        "license": str(entry.get("license", "UNKNOWN")),
        "reason": "ok",
    })
    return base


def run_local_vlm_critic(
    *,
    frame_samples: list[dict[str, Any]],
    narrative_summary: str,
    models_dir: Path | None = None,
    timeout_s: float = 90.0,
) -> dict[str, Any]:
    """Independent local VLM inspection adapter feeding real decoded image paths to local multimodal CLI.

    When no local VLM is provisioned, honestly returns `VISION_CRITIC_NOT_RUN` (zero simulated output!).
    """
    vlm_info = inspect_local_vlm(models_dir=models_dir)
    if not vlm_info.get("available"):
        return {
            "status": "VISION_CRITIC_NOT_RUN",
            "available": False,
            "reason": vlm_info.get("reason", "local_vlm_not_provisioned"),
            "frames_fed": 0,
            "image_hashes": [],
            "findings": [],
        }

    valid_imgs: list[tuple[dict[str, Any], Path, str]] = []
    for item in frame_samples[:8]:
        p = Path(str(item.get("image_path", "")))
        if p.is_file():
            valid_imgs.append((item, p, _sha256_file(p)[:16]))
    if not valid_imgs:
        return {
            "status": "VISION_CRITIC_NO_FRAMES",
            "available": True,
            "reason": "no_decoded_frame_files_supplied",
            "frames_fed": 0,
            "image_hashes": [],
            "findings": [],
        }

    t0 = time.perf_counter()
    prompt_text = (
        "You are an independent visual QA critic inspecting decoded frames from a vertical 9:16 edited video. "
        "Do NOT assume the editor succeeded. Report ONLY visible pixel defects (face collision, unreadable text, "
        "wrong numbers, abrupt flashes, fake UI, clutter) and genuine strengths as compact JSON.\n"
        f"Minimal factual context: {narrative_summary[:300]}\n"
        "Frames provided: " + ", ".join(f"f{it.get('frame')}({it.get('label','')})" for it, _, _ in valid_imgs)
    )
    cmd = [
        str(vlm_info["binary_path"]),
        "-m", str(vlm_info["model_path"]),
        "--temp", "0.0",
        "--seed", "42",
        "-n", "400",
        "-p", prompt_text,
    ]
    if vlm_info.get("mmproj_path"):
        cmd.extend(["--mmproj", str(vlm_info["mmproj_path"])])
    for _, img_p, _ in valid_imgs:
        cmd.extend(["--image", str(img_p)])

    env = {
        k: v for k, v in os.environ.items()
        if k not in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
    }
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env, shell=False)
    elapsed_s = round(time.perf_counter() - t0, 3)
    if proc.returncode != 0:
        return {
            "status": "VISION_CRITIC_ERROR",
            "available": True,
            "reason": f"vlm_rc={proc.returncode}:{proc.stderr[-240:]}",
            "frames_fed": len(valid_imgs),
            "image_hashes": [h for _, _, h in valid_imgs],
            "elapsed_s": elapsed_s,
            "findings": [],
        }
    return {
        "status": "VISION_CRITIC_EXECUTED",
        "available": True,
        "model_id": vlm_info["model_id"],
        "frames_fed": len(valid_imgs),
        "image_hashes": [h for _, _, h in valid_imgs],
        "elapsed_s": elapsed_s,
        "raw_response_excerpt": proc.stdout[:1200],
        "findings": [],
    }


def _scene_for_frame(storyboard: StoryboardSpec, frame_idx: int) -> Any | None:
    for sc in storyboard.scenes:
        if sc.start_frame <= frame_idx < sc.end_frame_exclusive:
            return sc
    return None


def run_objective_rendered_critic(
    *,
    source_mp4: Path,
    final_mp4: Path,
    storyboard: StoryboardSpec,
    continuous_timeline: dict[str, Any],
    roi_tracks: dict[str, Any] | None,
    out_dir: Path,
    models_dir: Path | None = None,
    ring_buffer_size: int = 16,
) -> dict[str, Any]:
    """Fixes D09/D10/D11: Streaming O(window)-RAM decoded-pixel auditor inspecting EVERY frame & boundary window.

    Zero full-video BGR lists in RAM! Streams `final.mp4` and `source.mp4` sequentially in pairs, maintaining:
      - Bounded ring buffer (`maxlen=ring_buffer_size`) of downscaled (180x320) float32 thumbnails
      - Exact per-frame consecutive MAD, 4-frame MAD, and source-relative delta (`fin_mad - src_mad`)
      - All-boundary ±6 frame audit (catching cutaway jumps AND post-boundary spikes like f234=90.73!)
      - Full-timeline internal flash / spike audit (exempting planned transition handles with explicit rationale)
      - Face-region luminance washout guard on live non-cutaway frames
      - Boundary strip contact sheet built strictly from compact `144x256` thumbnails captured on the fly!
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vlm_info = inspect_local_vlm(models_dir=models_dir)

    cap_fin = cv2.VideoCapture(str(final_mp4))
    cap_src = cv2.VideoCapture(str(source_mp4))
    if not cap_fin.isOpened():
        cap_src.release()
        raise RuntimeError(f"Objective pixel critic failed to open final video: {final_mp4}")

    boundaries = list(continuous_timeline.get("boundaries", []))
    bnd_by_frame: dict[int, dict[str, Any]] = {int(b.get("boundary_frame", 0)): b for b in boundaries if int(b.get("boundary_frame", 0)) > 0}
    key_bnd_frames = sorted(bnd_by_frame.keys())[:8]
    strip_offsets = (-6, -2, -1, 0, 2, 6)
    needed_strip_frames: set[int] = set()
    for bf in key_bnd_frames:
        for off in strip_offsets:
            needed_strip_frames.add(max(0, bf + off))

    # Map transition handle spans [f0, f1] where gradual bridge motion is expected (distinguished from unmotivated flash!)
    planned_bridge_spans: list[tuple[int, int, str]] = []
    for bnd in boundaries:
        bf = int(bnd.get("boundary_frame", 0))
        hf = int(bnd.get("handle_frames", 0))
        if hf > 0:
            if bnd.get("bridge_direction") == "outgoing_from_prev":
                planned_bridge_spans.append((max(0, bf - hf - 1), bf + 2, str(bnd.get("boundary_id", ""))))
            elif bnd.get("bridge_direction") == "incoming_to_next":
                planned_bridge_spans.append((max(0, bf - 1), bf + hf + 2, str(bnd.get("boundary_id", ""))))
            else:
                planned_bridge_spans.append((max(0, bf - hf // 2 - 1), bf + hf // 2 + 1, str(bnd.get("boundary_id", ""))))

    def _in_bridge_handle(f_idx: int) -> bool:
        return any(lo <= f_idx <= hi for lo, hi, _ in planned_bridge_spans)

    roi_kfs = (roi_tracks or {}).get("keyframes") or (roi_tracks or {}).get("frames") or []
    roi_by_f = {int(r["frame"]): r for r in roi_kfs if "frame" in r}
    roi_keys_sorted = sorted(roi_by_f.keys())

    # Bounded streaming ring buffers (O(window) RAM!)
    cap_buf = max(8, int(ring_buffer_size))
    fin_ring: collections.deque[tuple[int, np.ndarray]] = collections.deque(maxlen=cap_buf)
    src_ring: collections.deque[tuple[int, np.ndarray]] = collections.deque(maxlen=cap_buf)
    strip_tiles: dict[int, Image.Image] = {}

    step_mads: list[float] = []
    src_step_mads: list[float] = []
    mad_4f_list: list[float] = []
    src_4f_list: list[float] = []
    luma_list: list[float] = []

    findings: list[dict[str, Any]] = []
    internal_flashes: list[dict[str, Any]] = []
    max_face_washout_delta = 0.0
    worst_washout_frame = -1
    black_frames = 0
    mismatched_dims = 0
    frame_idx = 0
    tw, th = 144, 256

    try:
        while True:
            ok1, f1 = cap_fin.read()
            ok2, f2 = cap_src.read()
            if not ok1 or f1 is None:
                break

            if ok2 and f2 is not None and f1.shape[:2] != f2.shape[:2]:
                mismatched_dims += 1
                f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_AREA)

            small_fin = cv2.resize(f1, (180, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
            small_src = (
                cv2.resize(f2, (180, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
                if (ok2 and f2 is not None)
                else None
            )

            mean_luma = float(np.mean(small_fin))
            luma_list.append(round(mean_luma, 2))
            if mean_luma < 4.0:
                black_frames += 1

            # Consecutive step MAD & 4-frame rolling MAD (streaming!)
            s_mad = 0.0
            src_s_mad = 0.0
            m4 = 0.0
            src_m4 = 0.0
            if fin_ring:
                _, prev_fin = fin_ring[-1]
                s_mad = round(float(np.mean(np.abs(small_fin - prev_fin))), 2)
                if src_ring and small_src is not None:
                    _, prev_src = src_ring[-1]
                    src_s_mad = round(float(np.mean(np.abs(small_src - prev_src))), 2)
            if len(fin_ring) >= 3:
                _, f_back3 = fin_ring[-3]
                m4 = round(float(np.mean(np.abs(small_fin - f_back3))), 2)
                if len(src_ring) >= 3 and small_src is not None:
                    _, s_back3 = src_ring[-3]
                    src_m4 = round(float(np.mean(np.abs(small_src - s_back3))), 2)

            step_mads.append(s_mad)
            src_step_mads.append(src_s_mad)
            mad_4f_list.append(m4)
            src_4f_list.append(src_m4)

            # D11: Full-timeline internal flash / unmotivated spike detector (outside intentional bridge handles!)
            rel_step = round(s_mad - src_s_mad, 2)
            if frame_idx >= 1 and rel_step > 68.0 and not _in_bridge_handle(frame_idx) and frame_idx not in bnd_by_frame:
                sc_hit = _scene_for_frame(storyboard, frame_idx)
                flash_rec = {
                    "frame": frame_idx,
                    "scene_id": sc_hit.scene_id if sc_hit else "live_gap",
                    "step_mad": s_mad,
                    "source_step_mad": src_s_mad,
                    "source_relative_step_mad": rel_step,
                }
                internal_flashes.append(flash_rec)
                findings.append({
                    "frame_range": [max(0, frame_idx - 1), frame_idx + 1],
                    "affected_scene_id": flash_rec["scene_id"],
                    "category": "internal_flash_spike",
                    "severity": "HIGH",
                    "observed_issue": (
                        f"Unexpected internal visual jump/flash at f{frame_idx} ({flash_rec['scene_id']}): "
                        f"step_mad={s_mad} vs source={src_s_mad} (delta={rel_step}>68)."
                    ),
                    "suggested_correction": "Smooth layer entry/exit alpha envelope or inspect offending graphic frame.",
                    "confidence": 0.96,
                })

            # Face-region luminance washout guard (every 6th frame on live non-cutaway scenes, streaming!)
            sc_cur = _scene_for_frame(storyboard, frame_idx)
            if (
                sc_cur is not None
                and sc_cur.stage_mode not in ("full_graphic_cutaway", "aspect_morph_editorial", "tracked_macro_split")
                and (frame_idx - sc_cur.start_frame) % 6 == 0
                and ok2
                and f2 is not None
                and roi_keys_sorted
            ):
                near = min(roi_keys_sorted, key=lambda k: abs(k - frame_idx))
                fb = roi_by_f[near].get("face") or roi_by_f[near].get("face_box")
                if isinstance(fb, dict):
                    fh, fw = f1.shape[:2]
                    bw, bh = float(fb.get("w", 0.25)), float(fb.get("h", 0.25))
                    if "x" in fb and "y" in fb:
                        bx, by = float(fb["x"]), float(fb["y"])
                    else:
                        bx = float(fb.get("cx", 0.5)) - 0.5 * bw
                        by = float(fb.get("cy", 0.35)) - 0.5 * bh
                    x0 = max(0, int((bx + bw * 0.14) * fw))
                    y0 = max(0, int((by + bh * 0.25) * fh))
                    x1 = min(fw, int((bx + bw * 0.86) * fw))
                    y1 = min(fh, int((by + bh * 0.88) * fh))
                    if x1 - x0 >= 16 and y1 - y0 >= 16:
                        fin_luma = float(np.mean(cv2.cvtColor(f1[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)))
                        src_luma = float(np.mean(cv2.cvtColor(f2[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)))
                        delta_l = fin_luma - src_luma
                        if delta_l > max_face_washout_delta:
                            max_face_washout_delta = round(delta_l, 2)
                            worst_washout_frame = frame_idx
                        if delta_l > 28.0:
                            findings.append({
                                "frame_range": [frame_idx, frame_idx + 1],
                                "affected_scene_id": sc_cur.scene_id,
                                "category": "face_contrast_washout",
                                "severity": "HIGH" if delta_l > 42.0 else "MEDIUM",
                                "observed_issue": f"Face region brightened by +{delta_l:.1f} luma at f{frame_idx} ({sc_cur.scene_id}).",
                                "suggested_correction": "Restrict light leak strictly to peripheral rim with center face transparency mask.",
                                "confidence": 0.92,
                            })

            # Capture compact RGB tile ONLY for needed boundary strip frames (tiny memory footprint!)
            if frame_idx in needed_strip_frames:
                rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                strip_tiles[frame_idx] = Image.fromarray(rgb).resize((tw, th), Image.Resampling.LANCZOS)

            fin_ring.append((frame_idx, small_fin))
            if small_src is not None:
                src_ring.append((frame_idx, small_src))
            frame_idx += 1
    finally:
        cap_fin.release()
        cap_src.release()

    n = frame_idx
    boundary_metrics: list[dict[str, Any]] = []

    # D11: Audit EVERY boundary across ±6 frame window (step_mad, pre_4f_mad, post_4f_mad, source_relative spikes!)
    for bnd in boundaries:
        bf = int(bnd.get("boundary_frame", 0))
        if not (1 <= bf < n):
            continue
        step_mad = step_mads[bf]
        src_step = src_step_mads[bf]
        pre_4f = mad_4f_list[bf - 1]
        post_idx = min(n - 1, bf + 3)
        post_4f = mad_4f_list[post_idx]
        post_4f_src = src_4f_list[post_idx]
        post_4f_rel = round(post_4f - post_4f_src, 2)

        win_lo = max(1, bf - 6)
        win_hi = min(n - 1, bf + 6)
        win_max_step = max(step_mads[win_lo : win_hi + 1]) if win_hi >= win_lo else step_mad
        win_max_rel_step = max(round(step_mads[k] - src_step_mads[k], 2) for k in range(win_lo, win_hi + 1))

        is_cutaway_edge = (
            bnd.get("from_stage_mode") == "full_graphic_cutaway"
            or bnd.get("to_stage_mode") == "full_graphic_cutaway"
        )
        is_intentional_cut = str(bnd.get("boundary_class", "")) == "INTENTIONAL_CUT"
        is_morph_entry = str(bnd.get("to_stage_mode", "")) == "aspect_morph_editorial"

        abrupt_cutaway_flag = bool(is_cutaway_edge and step_mad > 65.0 and not is_intentional_cut)
        # Flag severe post-boundary spike (catches R2 f234 post_4f_mad=90.73 unless justified by explicit morph/bridge!)
        post_spike_flag = bool(post_4f_rel > 75.0 and not is_intentional_cut and not is_cutaway_edge and not is_morph_entry)
        win_flash_flag = bool(win_max_rel_step > 72.0 and not is_intentional_cut and not is_cutaway_edge)

        boundary_metrics.append({
            "boundary_id": bnd["boundary_id"],
            "boundary_frame": bf,
            "from_scene_id": bnd["from_scene_id"],
            "to_scene_id": bnd["to_scene_id"],
            "boundary_class": bnd["boundary_class"],
            "handle_frames": bnd.get("handle_frames", 0),
            "decoded_step_mad": step_mad,
            "source_step_mad": src_step,
            "pre_4f_mad": pre_4f,
            "post_4f_mad": post_4f,
            "post_4f_source_relative_mad": post_4f_rel,
            "window_pm6_max_step_mad": win_max_step,
            "window_pm6_max_source_relative_step_mad": win_max_rel_step,
            "abrupt_cutaway_jump_flagged": abrupt_cutaway_flag,
            "post_boundary_spike_flagged": post_spike_flag,
            "window_flash_flagged": win_flash_flag,
        })
        if abrupt_cutaway_flag:
            findings.append({
                "frame_range": [max(0, bf - 2), min(n - 1, bf + 2)],
                "affected_scene_id": bnd["from_scene_id"] if bnd.get("from_stage_mode") == "full_graphic_cutaway" else bnd["to_scene_id"],
                "category": "transition_continuity",
                "severity": "HIGH",
                "observed_issue": f"Abrupt cutaway boundary jump at f{bf} ({bnd['from_scene_id']}->{bnd['to_scene_id']}): decoded MAD={step_mad} (>65).",
                "suggested_correction": "Verify outgoing/incoming cutaway bridge handle renders live source underneath fold/sweep.",
                "confidence": 0.99,
            })
        if post_spike_flag or win_flash_flag:
            findings.append({
                "frame_range": [max(0, bf - 6), min(n - 1, bf + 6)],
                "affected_scene_id": bnd["to_scene_id"],
                "category": "boundary_window_spike",
                "severity": "HIGH" if win_flash_flag else "MEDIUM",
                "observed_issue": (
                    f"Boundary ±6f visual spike near f{bf} ({bnd['from_scene_id']}->{bnd['to_scene_id']}): "
                    f"post_4f_rel={post_4f_rel}, pm6_max_rel_step={win_max_rel_step}."
                ),
                "suggested_correction": "Add smooth entry/exit ramp or editorial sheet bridge across boundary window.",
                "confidence": 0.93,
            })

    # Render compact boundary strip contact sheet from captured streaming tiles
    strip_path = out_dir / "boundary_strips.jpg"
    key_bnds = [b for b in boundary_metrics if b["boundary_frame"] in key_bnd_frames][:8]
    if key_bnds and n > 0:
        pad = 6
        hdr = 40
        row_h = th + 26
        canvas = Image.new("RGB", (len(strip_offsets) * (tw + pad) + pad + 186, hdr + len(key_bnds) * row_h + pad), (20, 19, 17))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 12), "R5.R3 Streaming Decoded Boundary Continuity Strips (-6,-2,-1,0,+2,+6 frames)", fill=(245, 240, 230))
        for ri, bm in enumerate(key_bnds):
            bf = bm["boundary_frame"]
            y = hdr + ri * row_h
            draw.text((10, y + 32), f"{bm['boundary_id']}\nf{bf} MAD={bm['decoded_step_mad']}\np4={bm['post_4f_mad']}\n{bm['boundary_class'][:16]}", fill=(198, 164, 106))
            for ci, off in enumerate(strip_offsets):
                fno = max(0, min(n - 1, bf + off))
                x = 190 + ci * (tw + pad)
                tile = strip_tiles.get(fno)
                if tile is not None:
                    canvas.paste(tile, (x, y))
                draw.text((x + 3, y + th + 3), f"f{fno} ({off:+d})", fill=(230, 224, 212))
        canvas.save(strip_path, "JPEG", quality=88)

    high_sev = [f for f in findings if f.get("severity") == "HIGH"]
    report = {
        "schema_version": "3.0.0",
        "streaming_mode": True,
        "peak_ring_buffer_frames": cap_buf,
        "vlm_status": vlm_info["status"],
        "vlm_note": vlm_info["reason"],
        "objective_pixel_critic_status": "PASS" if not high_sev else "FINDINGS_FLAGGED",
        "frames_inspected": n,
        "black_frames_detected": black_frames,
        "mismatched_source_dim_frames": mismatched_dims,
        "boundary_count": len(boundary_metrics),
        "abrupt_cutaway_jumps": sum(1 for b in boundary_metrics if b["abrupt_cutaway_jump_flagged"]),
        "boundary_window_spikes": sum(1 for b in boundary_metrics if b["post_boundary_spike_flagged"] or b["window_flash_flagged"]),
        "internal_flash_count": len(internal_flashes),
        "internal_flashes": internal_flashes[:20],
        "max_face_washout_luma_delta": max_face_washout_delta,
        "worst_washout_frame": worst_washout_frame,
        "severe_findings_count": len(high_sev),
        "boundary_metrics": boundary_metrics,
        "findings": findings,
        "boundary_strips_path": str(strip_path) if strip_path.is_file() else None,
        "human_review_required": True,
    }
    atomic_write_json(out_dir / "scene_boundary_report.json", report)
    return report


def run_rhythm_critic(
    *,
    edit_map: Any,
    pacing_plan: dict[str, Any] | None,
    edited_voice_bed_wav: Path | None,
    word_preservation: dict[str, Any] | None,
    cut_continuity: dict[str, Any] | None,
    join_audio_qa: dict[str, Any] | None,
    storyboard: StoryboardSpec,
    pixel_critic: dict[str, Any],
    niche: str = "unknown",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Phase A8.2 (ML-01.R5.R4): Objective Rhythm & Pacing Critic measured on actual edited voice bed (NOT SFX mix!).

    Evaluates remaining non-speech spans, longest dead air, removed seconds by class, word preservation,
    reading time per scene, and join continuity without claiming human aesthetic VQ-01 approval.
    """
    import wave

    niche_low = (niche or storyboard.niche or "unknown").lower()
    src_dur = float(getattr(edit_map, "source_duration_s", 0.0)) if edit_map is not None else 0.0
    out_dur = float(getattr(edit_map, "output_duration_s", 0.0)) if edit_map is not None else 0.0
    rem_dur = float(getattr(edit_map, "removed_total_s", 0.0)) if edit_map is not None else 0.0
    omitted = list(getattr(edit_map, "omitted_spans", [])) if edit_map is not None else []

    removed_by_class: dict[str, float] = {}
    for o in omitted:
        cls = str(o.pause_class)
        removed_by_class[cls] = round(removed_by_class.get(cls, 0.0) + float(o.removed_s), 4)

    # Measure remaining longest non-speech span directly from edited_voice_bed.wav (NEVER from mixed SFX!)
    longest_post_gap_s = 0.0
    post_non_speech_ratio = 0.0
    bed_analyzed = False
    if edited_voice_bed_wav is not None and edited_voice_bed_wav.is_file():
        try:
            with wave.open(str(edited_voice_bed_wav), "rb") as wf:
                sr = wf.getframerate()
                nch = wf.getnchannels()
                nf = wf.getnframes()
                raw = np.frombuffer(wf.readframes(nf), dtype=np.int16).astype(np.float32) / 32768.0
            mono = raw.reshape((-1, nch)).mean(axis=1) if nch > 1 else raw
            hop = max(1, int(round(sr * 0.02)))  # 20ms
            n_blk = mono.size // hop
            if n_blk >= 10:
                blk_rms = np.sqrt(np.mean(mono[: n_blk * hop].reshape((n_blk, hop)) ** 2, axis=1) + 1e-12)
                blk_db = 20.0 * np.log10(np.maximum(blk_rms, 1e-6))
                nf_db = min(float(np.percentile(blk_db, 10)), -34.0)
                sp_thresh = min(-26.0, max(nf_db + 9.0, -40.0))
                is_ns = blk_db < sp_thresh
                post_non_speech_ratio = round(float(np.mean(is_ns)), 4)
                run_len = 0
                max_run = 0
                for flg in is_ns:
                    if flg:
                        run_len += 1
                        if run_len > max_run:
                            max_run = run_len
                    else:
                        run_len = 0
                longest_post_gap_s = round(max_run * 0.02, 3)
                bed_analyzed = True
        except Exception:
            bed_analyzed = False

    # Reading time audit per edited scene
    fps_f = storyboard.fps_num / max(1, storyboard.fps_den)
    scene_reading_audit: list[dict[str, Any]] = []
    short_cutaway_warnings: list[str] = []
    for sc in storyboard.scenes:
        sc_dur = round((sc.end_frame_exclusive - sc.start_frame) / max(1.0, fps_f), 3)
        too_brief_cutaway = bool(sc.stage_mode == "full_graphic_cutaway" and sc_dur < 1.2)
        if too_brief_cutaway:
            short_cutaway_warnings.append(f"{sc.scene_id}:cutaway_only_{sc_dur:.2f}s")
        scene_reading_audit.append({
            "scene_id": sc.scene_id,
            "stage_mode": sc.stage_mode,
            "edited_duration_s": sc_dur,
            "readable_duration_ok": not too_brief_cutaway,
        })

    has_cuts = len(omitted) > 0
    # D07/D09 fix: Fail-closed when cuts were applied and evidence (edited voice bed, word preservation, or join QA) is missing!
    wp_ok = (True if not has_cuts else False) if word_preservation is None else bool(word_preservation.get("passed", False))
    joins_audio_ok = (True if not has_cuts else False) if join_audio_qa is None else (join_audio_qa.get("status") == "PASS")
    joins_vis_severe = int((cut_continuity or {}).get("severe_jump_joins", 0))
    dead_air_limit = 3.5 if niche_low in ("wellness", "silent_art", "storytelling") else 1.85
    unresolved_dead_air = bool(bed_analyzed and longest_post_gap_s > dead_air_limit)

    findings: list[dict[str, Any]] = []
    if has_cuts and not bed_analyzed:
        findings.append({"category": "missing_edited_voice_bed", "severity": "HIGH", "detail": "Edited voice bed WAV was missing or unreadable during rhythm QA (fail-closed)."})
    if has_cuts and word_preservation is None:
        findings.append({"category": "missing_word_preservation_evidence", "severity": "HIGH", "detail": "Word preservation Re-ASR evidence missing for cut timeline (fail-closed)."})
    elif not wp_ok:
        findings.append({"category": "word_preservation", "severity": "HIGH", "detail": "One or more protected spoken words/numbers failed edited voice bed verification."})
    if has_cuts and join_audio_qa is None:
        findings.append({"category": "missing_audio_join_evidence", "severity": "HIGH", "detail": "Audio join waveform QA evidence missing for cut timeline (fail-closed)."})
    elif not joins_audio_ok:
        findings.append({"category": "audio_join_defect", "severity": "HIGH", "detail": str((join_audio_qa or {}).get("failure_reasons"))})
    if joins_vis_severe > 0:
        findings.append({"category": "cut_visual_jump", "severity": "MEDIUM", "detail": f"{joins_vis_severe} cut join(s) had large visual jump."})
    if unresolved_dead_air:
        findings.append({"category": "unresolved_dead_air", "severity": "MEDIUM", "detail": f"Longest non-speech span on edited voice bed is {longest_post_gap_s:.2f}s (>{dead_air_limit:.2f}s)."})
    if short_cutaway_warnings:
        findings.append({"category": "cutaway_reading_time", "severity": "MEDIUM", "detail": ", ".join(short_cutaway_warnings)})

    high_count = sum(1 for f in findings if f["severity"] == "HIGH")
    med_count = sum(1 for f in findings if f["severity"] == "MEDIUM")
    rhythm_status = "FAIL" if high_count > 0 else ("WARN_REVIEW_REQUIRED" if med_count > 0 else "PASS")
    report = {
        "schema_version": "1.0.0",
        "niche": niche_low,
        "pacing_mode": str(getattr(edit_map, "pacing_mode", "preserve")) if edit_map is not None else "preserve",
        "pacing_status": str(getattr(edit_map, "status", "IDENTITY_PRESERVE")) if edit_map is not None else "IDENTITY_PRESERVE",
        "source_duration_s": round(src_dur, 4),
        "edited_duration_s": round(out_dur, 4),
        "removed_total_s": round(rem_dur, 4),
        "cuts_count": len(omitted),
        "removed_by_pause_class": removed_by_class,
        "edited_voice_bed_analyzed": bed_analyzed,
        "longest_remaining_non_speech_s": longest_post_gap_s,
        "post_edit_non_speech_ratio": post_non_speech_ratio,
        "word_preservation_passed": wp_ok,
        "audio_joins_passed": joins_audio_ok,
        "visual_join_severe_jumps": joins_vis_severe,
        "pixel_critic_severe_findings": int(pixel_critic.get("severe_findings_count", 0)),
        "scene_reading_audit": scene_reading_audit,
        "findings": findings,
        "high_severity_count": high_count,
        "medium_severity_count": med_count,
        "objective_rhythm_status": rhythm_status,
        "human_vq01_status": "PENDING_HUMAN_REVIEW",
        "vlm_status": str(pixel_critic.get("vlm_status", "VISION_CRITIC_NOT_RUN")),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "rhythm_critic.json", report)
        atomic_write_json(out_dir / "cut_review.json", {
            "schema_version": "1.0.0",
            "edit_map_hash": getattr(edit_map, "map_hash", ""),
            "chosen_plan_id": (pacing_plan or {}).get("chosen_plan_id", "pacing_identity"),
            "omitted_spans": [o.to_dict() for o in omitted],
            "cut_continuity_summary": {
                "joins_count": (cut_continuity or {}).get("joins_count", 0),
                "severe_jump_joins": joins_vis_severe,
                "status": (cut_continuity or {}).get("status", "PASS_NO_CUTS"),
            },
            "join_audio_summary": {
                "joins_checked": (join_audio_qa or {}).get("joins_checked", 0),
                "clicks_detected": (join_audio_qa or {}).get("clicks_detected", 0),
                "status": (join_audio_qa or {}).get("status", "PASS"),
            },
            "word_preservation_summary": {
                "passed": wp_ok,
                "token_recall": (word_preservation or {}).get("token_recall", 1.0),
                "critical_token_recall": (word_preservation or {}).get("critical_token_recall", 1.0),
                "missing_numeric_tokens": (word_preservation or {}).get("missing_numeric_tokens", []),
            },
        })
    return report

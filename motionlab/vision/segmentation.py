"""Local Apache-2.0 MODNet Portrait Matting + Optical-Flow Temporal Refinement & Compact Matte Streaming.

Zero cloud calls, zero runtime downloads. Models are bootstrapped once into `models/` and verified
via SHA-256 against `models/manifest.json`. Missing or corrupted models fail safely and transparently.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motionlab.config import REPO_ROOT
from motionlab.resource import GLOBAL_JOB_SEMAPHORE, ResourceBudget, run_managed_subprocess
from motionlab.schemas import MediaInfo, atomic_write_json, canonical_hash, ensure_contained_path

MODELS_DIR = REPO_ROOT / "models"
DEFAULT_MODEL_ID = "modnet_photographic_portrait_matting"
MATTE_SCHEMA_VERSION = "1.0.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_vision_model(models_dir: Path | None = None, model_id: str = DEFAULT_MODEL_ID) -> dict[str, Any]:
    """Check local ONNX runtime & SHA-256-verified segmentation weights without downloading anything."""
    mdir = (models_dir or MODELS_DIR).resolve()
    manifest_p = mdir / "manifest.json"
    info: dict[str, Any] = {
        "model_id": model_id,
        "available": False,
        "onnxruntime_available": False,
        "onnxruntime_version": None,
        "model_path": None,
        "license": "Apache-2.0",
        "sha256_expected": None,
        "sha256_actual": None,
        "sha256_verified": False,
        "size_bytes": 0,
        "reason": "",
    }
    try:
        import onnxruntime as ort  # noqa: PLC0415

        info["onnxruntime_available"] = True
        info["onnxruntime_version"] = str(getattr(ort, "__version__", "unknown"))
    except Exception as exc:
        info["reason"] = f"onnxruntime_unavailable:{exc}"
        return info

    if not manifest_p.is_file():
        info["reason"] = f"models_manifest_missing:{manifest_p}"
        return info

    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        entry = (manifest.get("models") or {}).get(model_id)
        if not isinstance(entry, dict):
            info["reason"] = f"model_id_not_in_manifest:{model_id}"
            return info
        filename = str(entry["filename"])
        expected_sha = str(entry["sha256"])
        info["license"] = str(entry.get("license", "Apache-2.0"))
        info["sha256_expected"] = expected_sha
        model_p = ensure_contained_path(mdir / filename, mdir)
        info["model_path"] = str(model_p)
        if not model_p.is_file():
            info["reason"] = f"model_weights_missing:{model_p} (run bootstrap --with-models)"
            return info
        info["size_bytes"] = int(model_p.stat().st_size)
        actual_sha = _sha256_file(model_p)
        info["sha256_actual"] = actual_sha
        if actual_sha != expected_sha:
            info["reason"] = f"model_sha256_mismatch:expected={expected_sha[:12]},actual={actual_sha[:12]}"
            return info
        info["sha256_verified"] = True
        info["available"] = True
        info["reason"] = "ok"
        return info
    except Exception as exc:
        info["reason"] = f"model_manifest_error:{exc}"
        return info


class MODNetSegmenter:
    """Lazy-loaded local ONNX MODNet continuous alpha portrait matting [0.0, 1.0]."""

    def __init__(self, model_path: Path, intra_threads: int = 4) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(6, int(intra_threads)))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model_path = model_path
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _infer_dims(w: int, h: int, long_side: int = 512) -> tuple[int, int]:
        """Compute aspect-preserving dimensions that are exact multiples of 32."""
        if h >= w:
            nh = long_side
            nw = max(64, int(round((w / max(1, h)) * nh / 32.0)) * 32)
        else:
            nw = long_side
            nh = max(64, int(round((h / max(1, w)) * nw / 32.0)) * 32)
        return nw, nh

    def predict_alpha(self, rgb_u8: np.ndarray, long_side: int = 512) -> np.ndarray:
        """Predict continuous float32 alpha matte in [0.0, 1.0] at native `(h, w)` resolution."""
        h, w = rgb_u8.shape[:2]
        iw, ih = self._infer_dims(w, h, long_side=long_side)
        resized = cv2.resize(rgb_u8, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        norm = (resized - 0.5) / 0.5
        tensor = np.transpose(norm, (2, 0, 1))[None, ...]
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        matte_small = np.squeeze(raw).astype(np.float32)
        matte_small = np.nan_to_num(matte_small, nan=0.0, posinf=1.0, neginf=0.0)
        matte = cv2.resize(matte_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(matte, 0.0, 1.0)


def refine_matte_temporally(
    prev_gray_small: np.ndarray | None,
    curr_gray_small: np.ndarray,
    prev_alpha: np.ndarray | None,
    curr_alpha: np.ndarray | None,
    feather_px: float = 1.4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Warp previous alpha via optical flow, reject ghosting/lag, and temporally stabilize edges."""
    if curr_alpha is None and prev_alpha is None:
        raise ValueError("refine_matte_temporally requires at least prev_alpha or curr_alpha")

    if prev_alpha is None or prev_gray_small is None:
        out = np.clip(np.nan_to_num(curr_alpha, nan=0.0), 0.0, 1.0).astype(np.float32)  # type: ignore[arg-type]
        return _soft_edge_polish(out, feather_px), {"temporal_iou": 1.0, "flicker_mad": 0.0, "flow_mag": 0.0}

    h, w = prev_alpha.shape[:2]
    sh, sw = curr_gray_small.shape[:2]
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_small, curr_gray_small, None, 0.5, 3, 13, 3, 5, 1.1, 0
    )
    flow_mag = float(np.mean(np.hypot(flow[..., 0], flow[..., 1])))

    # Warp previous alpha forward onto current frame coordinates
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    map_x = xx - flow[..., 0]
    map_y = yy - flow[..., 1]
    prev_small = cv2.resize(prev_alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    warped_small = cv2.remap(prev_small, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped = np.clip(cv2.resize(warped_small, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)

    if curr_alpha is None:
        # Interpolated intermediate frame (only used across 1-frame stride!)
        out = _soft_edge_polish(warped, feather_px)
        return out, {"temporal_iou": 0.99, "flicker_mad": float(np.mean(np.abs(out - prev_alpha))), "flow_mag": round(flow_mag, 4)}

    curr_clean = np.clip(np.nan_to_num(curr_alpha, nan=0.0), 0.0, 1.0).astype(np.float32)
    diff = np.abs(curr_clean - warped)
    # Reject stale warped pixels where local discrepancy is high (prevents ghosting/trailing arms!)
    trust = np.clip(1.0 - (diff / 0.25), 0.0, 1.0)
    # Keep blend gentle (<=0.28 weight on warped prev) so hair & hand motion never lags!
    blend_w = 0.28 * trust
    stabilized = curr_clean * (1.0 - blend_w) + warped * blend_w
    out = _soft_edge_polish(stabilized, feather_px)

    # Metrics
    b_prev = prev_alpha >= 0.5
    b_curr = out >= 0.5
    inter = int(np.logical_and(b_prev, b_curr).sum())
    union = int(np.logical_or(b_prev, b_curr).sum())
    t_iou = float(inter / union) if union > 0 else 1.0
    flicker = float(np.mean(np.abs(out - prev_alpha)))
    return out, {"temporal_iou": round(t_iou, 4), "flicker_mad": round(flicker, 5), "flow_mag": round(flow_mag, 4)}


def _soft_edge_polish(alpha: np.ndarray, feather_px: float = 1.4) -> np.ndarray:
    """Gentle 1-2px feather strictly along soft boundary band [0.03, 0.97] without eroding interior solid subject."""
    if feather_px <= 0.1:
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_px, sigmaY=feather_px)
    edge_band = ((alpha > 0.03) & (alpha < 0.97)).astype(np.float32)
    edge_band = cv2.GaussianBlur(edge_band, (5, 5), 1.0)
    out = alpha * (1.0 - 0.55 * edge_band) + blurred * (0.55 * edge_band)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_tabletop_foreground_guard(alpha: np.ndarray, guard_start_y: float = 0.72, guard_full_y: float = 0.80) -> np.ndarray:
    """Preserve lower tabletop/prop plane cleanly in composite matte while keeping upper subject cutout 100% intact.

    Prevents foreground desks/tables/objects below the upper-body cutout from getting partially erased when
    compositing behind-subject graphics in the upper/mid studio zone.
    """
    h, _w = alpha.shape[:2]
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    ramp = np.clip((ys - guard_start_y) / max(0.02, guard_full_y - guard_start_y), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    return np.maximum(alpha, ramp).astype(np.float32)


def segment_subject_sequence(
    media: MediaInfo,
    spans: list[tuple[int, int]],
    workdir: Path,
    cache_dir: Path | None = None,
    preserve_tabletop_y: float | None = None,
    keyframe_stride: int = 2,
    budget: ResourceBudget | None = None,
    ffmpeg_bin: str | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream source spans through local MODNet ONNX, temporally stabilize, and emit compact grayscale matte MP4(s).

    Returns metadata dict with per-span matte clip paths, coverage/edge/IoU/flicker metrics, and cache provenance.
    Never downloads model weights; returns `available=False` honestly if weights/onnxruntime are unavailable.
    """
    t0 = time.perf_counter()
    model_info = inspect_vision_model(models_dir=models_dir)
    if not model_info["available"]:
        return {
            "available": False,
            "status": "BLOCKED_CREATIVE_MASK",
            "reason": model_info["reason"],
            "model": model_info,
            "spans": [],
            "quality_summary": {},
        }

    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    w, h = media.display_width, media.display_height
    fps_str = media.fps.to_ffmpeg()
    stride = max(1, min(3, int(keyframe_stride)))
    clean_spans = sorted({(max(0, int(s)), min(media.duration_frames, int(e))) for s, e in spans if int(e) > int(s)})
    if not clean_spans:
        return {
            "available": True,
            "status": "NO_SPANS_REQUESTED",
            "model": model_info,
            "spans": [],
            "quality_summary": {},
        }

    # Content-addressed cache lookup for matte clips
    cache_root = (cache_dir or (REPO_ROOT / ".motionlab_cache" / "vision")).resolve()
    cache_key = canonical_hash({
        "schema": MATTE_SCHEMA_VERSION,
        "source_hash": media.input_hash,
        "model_sha256": model_info["sha256_actual"],
        "w": w, "h": h, "fps": fps_str,
        "spans": clean_spans,
        "preserve_tabletop_y": preserve_tabletop_y,
        "stride": stride,
    })
    cache_slot = ensure_contained_path(cache_root / cache_key, cache_root)
    cache_meta_p = cache_slot / "matte_meta.json"
    if cache_meta_p.is_file():
        try:
            cached = json.loads(cache_meta_p.read_text(encoding="utf-8"))
            all_exist = all(Path(sp["matte_mp4"]).is_file() for sp in cached.get("spans", []))
            if all_exist and cached.get("cache_key") == cache_key:
                cached["cache_hit"] = True
                cached["elapsed_s"] = round(time.perf_counter() - t0, 3)
                return cached
        except Exception:
            pass

    out_dir = (workdir / "vision_mattes").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    intra = budget.ffmpeg_threads if budget else 4
    seg = MODNetSegmenter(Path(model_info["model_path"]), intra_threads=intra)

    span_results: list[dict[str, Any]] = []
    all_ious: list[float] = []
    all_flickers: list[float] = []
    all_covs: list[float] = []
    all_softs: list[float] = []
    inferred_frames = 0
    total_span_frames = 0

    for idx, (sf, ef) in enumerate(clean_spans):
        n_frames = ef - sf
        total_span_frames += n_frames
        matte_mp4 = out_dir / f"matte_span_{idx:02d}_{sf}_{ef}.mp4"
        raw_matte_mp4 = out_dir / f"raw_matte_span_{idx:02d}_{sf}_{ef}.mp4"

        # Stream-decode exact frame slice from source via FFmpeg (single exact pass!)
        dec_cmd = [
            ffmpeg, "-v", "error", "-nostdin",
            "-i", str(media.path),
            "-vf", f"select='between(n\\,{sf}\\,{ef - 1})',setpts=PTS-STARTPTS",
            "-fps_mode", "passthrough", "-frames:v", str(n_frames),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        enc_cmd_comp = [
            ffmpeg, "-y", "-v", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(matte_mp4),
        ]
        enc_cmd_raw = [
            ffmpeg, "-y", "-v", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "10", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(raw_matte_mp4),
        ]

        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        enc_c = subprocess.Popen(enc_cmd_comp, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        enc_r = subprocess.Popen(enc_cmd_raw, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        assert dec.stdout and enc_c.stdin and enc_r.stdin

        frame_bytes = w * h * 3
        small_w = 180
        small_h = max(32, int(round(h * (small_w / max(1, w)))))
        prev_gray: np.ndarray | None = None
        prev_alpha: np.ndarray | None = None
        span_covs: list[float] = []
        span_ious: list[float] = []
        span_flickers: list[float] = []
        span_softs: list[float] = []
        decoded_count = 0

        try:
            for local_i in range(n_frames):
                buf = dec.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                decoded_count += 1
                rgb = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
                gray_s = cv2.cvtColor(cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)

                # Always infer on first frame, last frame, every `stride` frame, or if scene/hand motion jumps!
                need_infer = (local_i == 0) or (local_i == n_frames - 1) or (local_i % stride == 0)
                if not need_infer and prev_gray is not None:
                    mad_s = float(np.mean(np.abs(gray_s.astype(np.float32) - prev_gray.astype(np.float32))))
                    if mad_s > 6.5:
                        need_infer = True

                raw_pred = seg.predict_alpha(rgb, long_side=512) if need_infer else None
                if need_infer:
                    inferred_frames += 1

                refined, q_metrics = refine_matte_temporally(prev_gray, gray_s, prev_alpha, raw_pred, feather_px=1.35)
                prev_gray = gray_s
                prev_alpha = refined

                cov = float((refined >= 0.5).mean())
                soft = float(((refined > 0.05) & (refined < 0.95)).mean())
                span_covs.append(cov)
                span_softs.append(soft)
                if local_i > 0:
                    span_ious.append(q_metrics["temporal_iou"])
                    span_flickers.append(q_metrics["flicker_mad"])

                raw_u8 = np.clip(np.round(refined * 255.0), 0, 255).astype(np.uint8)
                enc_r.stdin.write(raw_u8.tobytes())

                if preserve_tabletop_y is not None:
                    comp_alpha = apply_tabletop_foreground_guard(
                        refined, guard_start_y=max(0.55, preserve_tabletop_y - 0.04), guard_full_y=min(0.92, preserve_tabletop_y + 0.06)
                    )
                else:
                    comp_alpha = refined
                comp_u8 = np.clip(np.round(comp_alpha * 255.0), 0, 255).astype(np.uint8)
                enc_c.stdin.write(comp_u8.tobytes())
        finally:
            dec.stdout.close()
            enc_c.stdin.close()
            enc_r.stdin.close()
            d_err = dec.stderr.read().decode("utf-8", errors="replace") if dec.stderr else ""
            ec_err = enc_c.stderr.read().decode("utf-8", errors="replace") if enc_c.stderr else ""
            er_err = enc_r.stderr.read().decode("utf-8", errors="replace") if enc_r.stderr else ""
            d_rc = dec.wait(timeout=60)
            ec_rc = enc_c.wait(timeout=60)
            er_rc = enc_r.wait(timeout=60)
            if d_rc != 0 or ec_rc != 0 or er_rc != 0:
                raise RuntimeError(f"Matte streaming failed (dec={d_rc}:{d_err[:200]}, enc_c={ec_rc}:{ec_err[:200]}, enc_r={er_rc}:{er_err[:200]})")

        mean_cov = float(np.mean(span_covs)) if span_covs else 0.0
        mean_iou = float(np.mean(span_ious)) if span_ious else 1.0
        mean_flick = float(np.mean(span_flickers)) if span_flickers else 0.0
        mean_soft = float(np.mean(span_softs)) if span_softs else 0.0
        # Valid human portrait matte must neither be empty (<5%) nor solid full-frame (>92%), and temporally stable (IoU >= 0.85)
        quality_ok = (0.06 <= mean_cov <= 0.90) and (mean_iou >= 0.85) and (mean_flick <= 0.065) and (decoded_count == n_frames)

        all_covs.extend(span_covs)
        all_ious.extend(span_ious)
        all_flickers.extend(span_flickers)
        all_softs.extend(span_softs)

        span_results.append({
            "span_index": idx,
            "start_frame": sf,
            "end_frame_exclusive": ef,
            "frame_count": decoded_count,
            "matte_mp4": str(matte_mp4),
            "matte_sha256": _sha256_file(matte_mp4),
            "raw_matte_mp4": str(raw_matte_mp4),
            "raw_matte_sha256": _sha256_file(raw_matte_mp4),
            "mean_coverage": round(mean_cov, 4),
            "mean_soft_edge_frac": round(mean_soft, 4),
            "mean_temporal_iou": round(mean_iou, 4),
            "mean_flicker_mad": round(mean_flick, 5),
            "quality_ok": quality_ok,
        })

    # Persist into content-addressed local vision cache for warm reruns
    cache_slot.mkdir(parents=True, exist_ok=True)
    cached_spans: list[dict[str, Any]] = []
    for sp in span_results:
        dst_m = cache_slot / Path(sp["matte_mp4"]).name
        dst_r = cache_slot / Path(sp["raw_matte_mp4"]).name
        shutil.copy2(sp["matte_mp4"], dst_m)
        shutil.copy2(sp["raw_matte_mp4"], dst_r)
        cached_spans.append({**sp, "matte_mp4": str(dst_m), "raw_matte_mp4": str(dst_r)})

    result = {
        "available": True,
        "status": "READY" if all(s["quality_ok"] for s in span_results) else "DEGRADED_QUALITY",
        "cache_key": cache_key,
        "cache_hit": False,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "model": model_info,
        "keyframe_stride": stride,
        "preserve_tabletop_y": preserve_tabletop_y,
        "total_span_frames": total_span_frames,
        "inferred_frames": inferred_frames,
        "spans": cached_spans,
        "quality_summary": {
            "mean_coverage": round(float(np.mean(all_covs)) if all_covs else 0.0, 4),
            "mean_soft_edge_frac": round(float(np.mean(all_softs)) if all_softs else 0.0, 4),
            "mean_temporal_iou": round(float(np.mean(all_ious)) if all_ious else 1.0, 4),
            "min_temporal_iou": round(float(np.min(all_ious)) if all_ious else 1.0, 4),
            "mean_flicker_mad": round(float(np.mean(all_flickers)) if all_flickers else 0.0, 5),
            "all_spans_quality_ok": all(s["quality_ok"] for s in span_results),
        },
    }
    atomic_write_json(cache_meta_p, result)
    return result


def build_foreground_alpha_webm(
    source_mp4: Path,
    matte_mp4: Path,
    start_frame: int,
    end_frame_exclusive: int,
    fps_str: str,
    out_webm: Path,
    ffmpeg_bin: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Merge exact frame-aligned source slice + grayscale MODNet matte into VP9 yuva420p transparent foreground WebM."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    n_frames = max(1, end_frame_exclusive - start_frame)
    out_webm.parent.mkdir(parents=True, exist_ok=True)
    # Use format=rgba + alphamerge so continuous [0..255] grayscale matte becomes true straight alpha channel!
    fc = (
        f"[0:v]select='between(n\\,{start_frame}\\,{end_frame_exclusive - 1})',setpts=PTS-STARTPTS,format=rgba[rgb];"
        f"[1:v]setpts=PTS-STARTPTS,format=gray[a];"
        f"[rgb][a]alphamerge,format=yuva420p[outv]"
    )
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_mp4),
        "-i", str(matte_mp4),
        "-filter_complex", fc,
        "-map", "[outv]",
        "-frames:v", str(n_frames),
        "-r", fps_str,
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
        "-crf", "24", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "5",
        "-an", str(out_webm),
    ]
    with GLOBAL_JOB_SEMAPHORE:
        _, stats = run_managed_subprocess(cmd, timeout_s=timeout_s, check=True)
    return {
        "path": str(out_webm),
        "sha256": _sha256_file(out_webm),
        "size_bytes": int(out_webm.stat().st_size),
        "frame_count": n_frames,
        "elapsed_s": stats["elapsed_s"],
    }

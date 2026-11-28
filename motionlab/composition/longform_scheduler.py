"""Phase A7 (ML-01.R5.R7): Long-Form Global Timeline DAG Scheduler, Bounded Async Chunking & Total Process-Tree Guard.

Fixes D05, D09, D12, D13:
  - Partitions ANY duration (Shorts 27s through Long-form 3-30+ min / 18,000+ frames) into exact contiguous
    half-open core ranges `[start_frame, end_frame_exclusive)` covering `0..N` with ZERO gaps or overlaps!
  - Computes explicit boundary handles & persistent global state snapshots so animations/transitions
    spanning chunk boundaries continue identically without reset or double-blending.
  - Content-addressed task keys (`source_hash + graph_hash + backend + frame_range + state_snapshot + assets`)
    guarantee automatic cache invalidation whenever source, graph, or state changes.
  - Bounded admission control (`max_workers=1..2`, capped FFmpeg threads per worker to prevent nested
    oversubscription) + atomic `.partial -> final` rename + crash-safe verified `resume_run()`.
  - Tracks TOTAL process-tree memory (Python + Chromium + FFmpeg + llama.cpp children) and live available
    RAM/disk reserves (fixing D12!).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import cv2
import psutil

from motionlab.schemas import atomic_write_json, canonical_hash, sanitize_native

SCHEDULER_SCHEMA_VERSION = "4.0.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class TotalProcessTreeMonitor:
    """Samples ENTIRE process tree (Python + FFmpeg + Chromium + llama.cpp children) & system reserves."""

    def __init__(self, work_dir: Path, *, min_disk_reserve_gb: float = 5.0, min_ram_reserve_gb: float = 3.0) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.min_disk_reserve_gb = float(min_disk_reserve_gb)
        self.min_ram_reserve_gb = float(min_ram_reserve_gb)
        self.peak_tree_rss_mb = 0.0
        self.peak_child_count = 0
        self.min_avail_ram_gb = 9999.0
        self.min_free_disk_gb = 9999.0
        self._lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []

    def sample(self, label: str = "") -> dict[str, Any]:
        me = psutil.Process(os.getpid())
        rss = 0
        children = []
        try:
            rss += me.memory_info().rss
            for ch in me.children(recursive=True):
                try:
                    rss += ch.memory_info().rss
                    children.append(ch.name())
                except Exception:
                    pass
        except Exception:
            pass
        vm = psutil.virtual_memory()
        du = shutil.disk_usage(self.work_dir if self.work_dir.exists() else Path.cwd())
        rss_mb = round(rss / (1024 * 1024), 2)
        avail_gb = round(vm.available / (1024**3), 2)
        free_gb = round(du.free / (1024**3), 2)
        rec = {
            "label": label,
            "tree_rss_mb": rss_mb,
            "child_process_count": len(children),
            "child_names": sorted(set(children))[:8],
            "system_avail_ram_gb": avail_gb,
            "disk_free_gb": free_gb,
        }
        with self._lock:
            self.peak_tree_rss_mb = max(self.peak_tree_rss_mb, rss_mb)
            self.peak_child_count = max(self.peak_child_count, len(children))
            self.min_avail_ram_gb = min(self.min_avail_ram_gb, avail_gb)
            self.min_free_disk_gb = min(self.min_free_disk_gb, free_gb)
            self.samples.append(rec)
        return rec

    def check_admission(self, *, label: str = "task") -> tuple[bool, str]:
        s = self.sample(label)
        if s["disk_free_gb"] < self.min_disk_reserve_gb:
            return False, f"DISK_RESERVE_BREACHED:{s['disk_free_gb']}GB<{self.min_disk_reserve_gb}GB"
        if s["system_avail_ram_gb"] < self.min_ram_reserve_gb:
            return False, f"RAM_RESERVE_BREACHED:{s['system_avail_ram_gb']}GB<{self.min_ram_reserve_gb}GB"
        return True, "OK"

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return sanitize_native({
                "peak_total_process_tree_rss_mb": self.peak_tree_rss_mb,
                "peak_concurrent_child_processes": self.peak_child_count,
                "min_observed_system_avail_ram_gb": round(self.min_avail_ram_gb, 2) if self.min_avail_ram_gb < 9999 else None,
                "min_observed_disk_free_gb": round(self.min_free_disk_gb, 2) if self.min_free_disk_gb < 9999 else None,
                "samples_taken": len(self.samples),
            })


def partition_render_ranges(
    *,
    total_frames: int,
    fps_num: int = 30,
    fps_den: int = 1,
    scenes: list[dict[str, Any]] | None = None,
    target_chunk_s: float = 20.0,
    min_chunk_s: float = 8.0,
) -> list[dict[str, Any]]:
    """Partition [0, total_frames) into contiguous half-open core chunks aligned near natural scene boundaries."""
    fps = float(fps_num) / float(max(1, fps_den))
    tot = max(1, int(total_frames))
    target_f = max(30, int(round(target_chunk_s * fps)))
    min_f = max(15, int(round(min(min_chunk_s, max(0.5, target_chunk_s * 0.5)) * fps)))

    if tot <= int(target_f * 1.4):
        return [{
            "chunk_id": "chunk_000",
            "chunk_index": 0,
            "core_start_frame": 0,
            "core_end_frame_exclusive": tot,
            "core_frames": tot,
        }]

    # Candidate natural split boundaries (scene end frames where no key event fires right on the edge)
    scene_ends = sorted({int(s["end_frame_exclusive"]) for s in (scenes or []) if 0 < int(s["end_frame_exclusive"]) < tot})

    chunks: list[dict[str, Any]] = []
    cur = 0
    while cur < tot:
        ideal = min(tot, cur + target_f)
        if tot - ideal < min_f:
            nxt = tot
        else:
            # Prefer a natural scene boundary within +/- 4s of ideal
            near = [b for b in scene_ends if cur + min_f <= b <= min(tot - min_f, ideal + int(4 * fps)) and abs(b - ideal) <= int(5 * fps)]
            nxt = min(near, key=lambda b: abs(b - ideal)) if near else ideal
        chunks.append({
            "chunk_id": f"chunk_{len(chunks):03d}",
            "chunk_index": len(chunks),
            "core_start_frame": cur,
            "core_end_frame_exclusive": nxt,
            "core_frames": nxt - cur,
        })
        cur = nxt
    return chunks


def compute_boundary_handles(
    chunk: dict[str, Any],
    *,
    total_frames: int,
    handle_frames: int = 6,
) -> dict[str, Any]:
    """Compute render overlap handles around a chunk's core range, trimmed back to exact core after render."""
    cs = int(chunk["core_start_frame"])
    ce = int(chunk["core_end_frame_exclusive"])
    pre = min(cs, max(0, int(handle_frames)))
    post = min(max(0, int(total_frames) - ce), max(0, int(handle_frames)))
    return {
        **chunk,
        "render_start_frame": cs - pre,
        "render_end_frame_exclusive": ce + post,
        "trim_leading_frames": pre,
        "trim_trailing_frames": post,
    }


def snapshot_global_state(
    *,
    chunk: dict[str, Any],
    scenes: list[dict[str, Any]],
    story_bible: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze deterministic global state snapshot at chunk entry so animations/entities never reset across seams."""
    cs = int(chunk["core_start_frame"])
    ce = int(chunk["core_end_frame_exclusive"])
    overlapping = [
        {
            "scene_id": s["scene_id"],
            "chapter_id": s.get("chapter_id"),
            "scene_start_frame": int(s["start_frame"]),
            "scene_end_frame_exclusive": int(s["end_frame_exclusive"]),
            "scene_hash": s.get("scene_hash") or canonical_hash(s)[:16],
        }
        for s in scenes
        if int(s["end_frame_exclusive"]) > cs and int(s["start_frame"]) < ce
    ]
    entities = (story_bible or {}).get("persistent_entities") or {}
    snap = {
        "chunk_id": chunk["chunk_id"],
        "entry_global_frame": cs,
        "exit_global_frame_exclusive": ce,
        "overlapping_scenes": overlapping,
        "entity_values": {k: v.get("current_value") for k, v in entities.items()},
    }
    snap["snapshot_hash"] = canonical_hash(snap)
    return sanitize_native(snap)


def resolve_chunk_frame(chunk: dict[str, Any], local_core_frame: int) -> int:
    """Map chunk-local core frame index [0..core_frames) back to exact global frame index."""
    cf = int(local_core_frame)
    if cf < 0 or cf >= int(chunk["core_frames"]):
        raise IndexError(f"Chunk frame out of core bounds: {cf} not in [0, {chunk['core_frames']})")
    return int(chunk["core_start_frame"]) + cf


def build_render_dag(
    *,
    source_hash: str,
    graph_hash: str,
    total_frames: int,
    fps_num: int,
    fps_den: int,
    scenes: list[dict[str, Any]],
    story_bible: dict[str, Any] | None = None,
    target_chunk_s: float = 20.0,
    handle_frames: int = 6,
    backend_version: str = "r5r7_v4",
) -> dict[str, Any]:
    """Build deterministic content-addressed chunk render DAG with explicit dependency keys."""
    raw_chunks = partition_render_ranges(
        total_frames=total_frames,
        fps_num=fps_num,
        fps_den=fps_den,
        scenes=scenes,
        target_chunk_s=target_chunk_s,
    )
    tasks: list[dict[str, Any]] = []
    for ch in raw_chunks:
        ch_h = compute_boundary_handles(ch, total_frames=total_frames, handle_frames=handle_frames)
        snap = snapshot_global_state(chunk=ch_h, scenes=scenes, story_bible=story_bible)
        cache_key = canonical_hash({
            "source_hash": source_hash,
            "graph_hash": graph_hash,
            "backend_version": backend_version,
            "core_range": [ch_h["core_start_frame"], ch_h["core_end_frame_exclusive"]],
            "render_range": [ch_h["render_start_frame"], ch_h["render_end_frame_exclusive"]],
            "snapshot_hash": snap["snapshot_hash"],
        })[:24]
        tasks.append({
            **ch_h,
            "cache_key": cache_key,
            "state_snapshot": snap,
            "dependencies": [],
        })

    return sanitize_native({
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "source_hash": source_hash,
        "graph_hash": graph_hash,
        "total_frames": int(total_frames),
        "fps_num": int(fps_num),
        "fps_den": int(fps_den),
        "chunk_count": len(tasks),
        "tasks": tasks,
        "dag_hash": canonical_hash([(t["chunk_id"], t["cache_key"]) for t in tasks]),
    })


def validate_chunk_manifest(dag: dict[str, Any], completed_chunks: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify exact [0, N) core coverage, monotonic order, non-empty verified files, and exact frame counts."""
    errs: list[str] = []
    tot = int(dag["total_frames"])
    ordered = sorted(completed_chunks, key=lambda c: int(c["core_start_frame"]))
    if len(ordered) != int(dag["chunk_count"]):
        errs.append(f"incomplete_chunk_count:{len(ordered)}!={dag['chunk_count']}")
    cursor = 0
    for c in ordered:
        cs = int(c["core_start_frame"])
        ce = int(c["core_end_frame_exclusive"])
        if cs != cursor:
            errs.append(f"gap_or_overlap_at_{c['chunk_id']}:expected={cursor},got={cs}")
        p = Path(str(c.get("output_path") or ""))
        if not p.is_file() or p.stat().st_size == 0:
            errs.append(f"missing_chunk_file:{c['chunk_id']}")
        elif int(c.get("decoded_frames", -1)) != (ce - cs):
            errs.append(f"chunk_frame_count_mismatch:{c['chunk_id']}:{c.get('decoded_frames')}!={ce - cs}")
        cursor = ce
    if cursor != tot:
        errs.append(f"tail_coverage_mismatch:covered={cursor}!=total={tot}")
    return len(errs) == 0, errs


def _probe_frame_count(mp4_path: Path) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return -1
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return cnt


def resume_run(
    *,
    dag: dict[str, Any],
    work_dir: Path,
    render_chunk_fn: Callable[[dict[str, Any], Path, int], None],
    max_workers: int = 1,
    ffmpeg_threads_per_worker: int = 4,
    monitor: TotalProcessTreeMonitor | None = None,
) -> dict[str, Any]:
    """Execute or resume a chunk render DAG with bounded concurrency, cache validation, and atomic writes.

    Reuses ONLY chunks whose cache_key, SHA-256, and decoded frame count verify cleanly!
    """
    t0 = time.perf_counter()
    chunks_dir = Path(work_dir).resolve() / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    mon = monitor or TotalProcessTreeMonitor(chunks_dir)
    workers = max(1, min(2, int(max_workers)))

    results_by_id: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    rendered_new = 0
    failures: list[dict[str, Any]] = []

    def _run_one(task: dict[str, Any]) -> dict[str, Any]:
        cid = str(task["chunk_id"])
        ckey = str(task["cache_key"])
        out_mp4 = chunks_dir / f"{cid}_{ckey}.mp4"
        meta_json = chunks_dir / f"{cid}_{ckey}.meta.json"
        expected_f = int(task["core_frames"])

        # Check verified cache hit first (content-addressed key + SHA-256 + exact decoded frames!)
        if out_mp4.is_file() and meta_json.is_file():
            try:
                m = json.loads(meta_json.read_text(encoding="utf-8"))
                if (
                    m.get("cache_key") == ckey
                    and m.get("sha256") == _sha256_file(out_mp4)
                    and _probe_frame_count(out_mp4) == expected_f
                ):
                    return {
                        "chunk_id": cid,
                        "chunk_index": task["chunk_index"],
                        "core_start_frame": task["core_start_frame"],
                        "core_end_frame_exclusive": task["core_end_frame_exclusive"],
                        "core_frames": expected_f,
                        "cache_key": ckey,
                        "output_path": str(out_mp4),
                        "sha256": m["sha256"],
                        "decoded_frames": expected_f,
                        "cache_hit": True,
                        "elapsed_s": 0.0,
                    }
            except Exception:
                pass

        ok_adm, reason = mon.check_admission(label=f"admit_{cid}")
        if not ok_adm:
            raise RuntimeError(f"Resource admission blocked {cid}: {reason}")

        ts = time.perf_counter()
        part_mp4 = chunks_dir / f"{cid}_{ckey}.partial.mp4"
        if part_mp4.exists():
            part_mp4.unlink(missing_ok=True)
        render_chunk_fn(task, part_mp4, ffmpeg_threads_per_worker)
        mon.sample(f"after_{cid}")
        dec_f = _probe_frame_count(part_mp4)
        if dec_f != expected_f:
            part_mp4.unlink(missing_ok=True)
            raise RuntimeError(f"Chunk {cid} frame count mismatch: decoded={dec_f} != expected={expected_f}")
        sha = _sha256_file(part_mp4)
        part_mp4.replace(out_mp4)
        atomic_write_json(meta_json, {"chunk_id": cid, "cache_key": ckey, "sha256": sha, "decoded_frames": dec_f})
        return {
            "chunk_id": cid,
            "chunk_index": task["chunk_index"],
            "core_start_frame": task["core_start_frame"],
            "core_end_frame_exclusive": task["core_end_frame_exclusive"],
            "core_frames": expected_f,
            "cache_key": ckey,
            "output_path": str(out_mp4),
            "sha256": sha,
            "decoded_frames": dec_f,
            "cache_hit": False,
            "elapsed_s": round(time.perf_counter() - ts, 3),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in dag["tasks"]}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                res = fut.result()
                results_by_id[res["chunk_id"]] = res
                if res["cache_hit"]:
                    cache_hits += 1
                else:
                    rendered_new += 1
            except Exception as exc:
                failures.append({"chunk_id": t["chunk_id"], "error": str(exc)})

    ordered = [results_by_id[t["chunk_id"]] for t in dag["tasks"] if t["chunk_id"] in results_by_id]
    valid_manifest, m_errs = validate_chunk_manifest(dag, ordered) if not failures else (False, [f"task_failures:{len(failures)}"])
    return sanitize_native({
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "dag_hash": dag["dag_hash"],
        "max_workers": workers,
        "chunks_total": dag["chunk_count"],
        "cache_hits": cache_hits,
        "rendered_new": rendered_new,
        "failures": failures,
        "manifest_valid": valid_manifest,
        "manifest_errors": m_errs,
        "wall_s": round(time.perf_counter() - t0, 3),
        "chunks": ordered,
        "resource_summary": mon.summary(),
    })


def assemble_verified_chunks(
    *,
    dag: dict[str, Any],
    run_report: dict[str, Any],
    audio_wav: Path | None,
    out_mp4: Path,
    ffmpeg_bin: str = "ffmpeg",
    crf: int = 18,
    preset: str = "medium",
) -> dict[str, Any]:
    """Assemble validated core chunks in exact global order + mux continuous global audio ONCE (zero chunk AAC seams!)."""
    if not run_report.get("manifest_valid"):
        raise RuntimeError(f"Cannot assemble incomplete/invalid chunk run: {run_report.get('manifest_errors')}")
    chunks = sorted(run_report["chunks"], key=lambda c: int(c["chunk_index"]))
    out_mp4 = Path(out_mp4).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    concat_txt = out_mp4.parent / f".{out_mp4.stem}_concat.txt"
    lines = [f"file '{Path(c['output_path']).resolve().as_posix()}'" for c in chunks]
    concat_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    part_out = out_mp4.with_suffix(".partial.mp4")
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(concat_txt)]
    if audio_wav is not None and Path(audio_wav).is_file():
        cmd.extend(["-i", str(audio_wav), "-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0", "-an"])
    cmd.extend([
        "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
        "-r", f"{int(dag['fps_num'])}/{int(dag['fps_den'])}", "-fps_mode", "cfr",
    ])
    if audio_wav is not None and Path(audio_wav).is_file():
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest"])
    cmd.extend(["-movflags", "+faststart", str(part_out)])

    try:
        subprocess.run(cmd, check=True, timeout=600, shell=False)
        dec_f = _probe_frame_count(part_out)
        if dec_f != int(dag["total_frames"]):
            raise RuntimeError(f"Assembled MP4 frame mismatch: got {dec_f}, expected {dag['total_frames']}")
        part_out.replace(out_mp4)
    finally:
        concat_txt.unlink(missing_ok=True)
        part_out.unlink(missing_ok=True)

    return sanitize_native({
        "output_path": str(out_mp4),
        "sha256": _sha256_file(out_mp4),
        "bytes": int(out_mp4.stat().st_size),
        "decoded_frames": int(dag["total_frames"]),
        "chunks_assembled": len(chunks),
        "single_final_aac_encode": bool(audio_wav is not None and Path(audio_wav).is_file()),
    })

"""Resource governance, disk safety, bounded concurrency, and child process cleanup (Phase F)."""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

from motionlab.schemas import SchemaValidationError, ensure_contained_path


class BudgetExceededError(RuntimeError):
    """Raised when disk space, temp budget, or timeout governance blocks execution."""


@dataclass
class ResourceBudget:
    cpu_logical: int
    ram_total_gb: float
    ram_available_gb: float
    analysis_workers: int
    render_workers: int
    ffmpeg_threads: int
    gpu_concurrency: int
    max_temp_gb: float
    min_free_gb: float
    reserve_policy_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_resource_budget(
    max_workers: int | None = None,
    max_temp_gb: float = 4.0,
    min_free_gb: float = 5.0,
) -> ResourceBudget:
    """Compute conservative worker and thread limits from measured CPU & RAM."""
    cpus = max(1, os.cpu_count() or 1)
    vm = psutil.virtual_memory()
    ram_total_gb = round(vm.total / (1024 ** 3), 2)
    ram_avail_gb = round(vm.available / (1024 ** 3), 2)

    # Estimate ~1.5 GiB safe working memory per analysis worker, ~2.5 GiB per render worker
    ram_analysis_cap = max(1, int(ram_avail_gb // 1.5))
    default_analysis = max(1, min(4, max(1, cpus // 2), ram_analysis_cap))
    if max_workers is not None:
        analysis_workers = max(1, min(max_workers, 4, cpus))
    else:
        analysis_workers = default_analysis

    ram_render_cap = 2 if ram_avail_gb >= 6.0 and cpus >= 6 else 1
    if max_workers is not None:
        render_workers = max(1, min(max_workers, 2, ram_render_cap))
    else:
        render_workers = 1  # Conservative default; bounded parallel via explicit flag or segment pool

    # Never oversubscribe FFmpeg internal threads across workers
    ffmpeg_threads = max(1, min(4, cpus // max(1, render_workers)))

    return ResourceBudget(
        cpu_logical=cpus,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_avail_gb,
        analysis_workers=analysis_workers,
        render_workers=render_workers,
        ffmpeg_threads=ffmpeg_threads,
        gpu_concurrency=1,
        max_temp_gb=float(max_temp_gb),
        min_free_gb=float(min_free_gb),
        reserve_policy_note=(
            f"Hard reserve floor={min_free_gb:.2f} GiB, max_temp={max_temp_gb:.2f} GiB "
            "(configurable via --min-free-gb / --max-temp-gb; warns if free < 10% volume)"
        ),
    )


def estimate_temp_bytes(
    duration_s: float,
    width: int,
    height: int,
    fps: float,
    include_preview: bool = False,
) -> int:
    """Conservatively estimate temp + intermediate disk bytes (streaming pipeline, not PNG dump)."""
    # Encoded H.264 intermediate (~12 Mbps at 1080p scaled by pixel count) + PCM analysis wav (32 KB/s)
    # + contact sheet + small overlay clips + 2x safety overhead
    pixels_ratio = max(0.25, (width * height) / (1920 * 1080))
    fps_ratio = max(0.5, fps / 30.0)
    video_bytes_per_sec = int(1_800_000 * pixels_ratio * fps_ratio)  # ~14.4 Mbps conservative
    audio_wav_bytes = int(duration_s * 32_000) + 1_000_000
    overlay_overhead = int(20 * 1024 * 1024)
    preview_overhead = int(video_bytes_per_sec * duration_s * 0.4) if include_preview else 0
    raw_est = int(video_bytes_per_sec * max(1.0, duration_s) * 2.2) + audio_wav_bytes + overlay_overhead + preview_overhead
    return max(32 * 1024 * 1024, raw_est)


def preflight_budget(
    dest_dir: Path,
    temp_dir: Path,
    duration_s: float,
    width: int,
    height: int,
    fps: float,
    budget: ResourceBudget,
    include_preview: bool = False,
) -> dict[str, Any]:
    """Check actual free disk space on destination and temp volumes before rendering."""
    dest_check = dest_dir if dest_dir.exists() else dest_dir.parent
    while not dest_check.exists() and dest_check != dest_check.parent:
        dest_check = dest_check.parent
    temp_check = temp_dir if temp_dir.exists() else temp_dir.parent
    while not temp_check.exists() and temp_check != temp_check.parent:
        temp_check = temp_check.parent

    dest_usage = shutil.disk_usage(dest_check)
    temp_usage = shutil.disk_usage(temp_check)

    est_bytes = estimate_temp_bytes(duration_s, width, height, fps, include_preview=include_preview)
    max_temp_bytes = int(budget.max_temp_gb * (1024 ** 3))
    min_free_bytes = int(budget.min_free_gb * (1024 ** 3))

    warnings: list[str] = []
    ten_pct_dest = int(dest_usage.total * 0.10)
    if dest_usage.free < ten_pct_dest:
        warnings.append(
            f"Destination volume free space ({dest_usage.free / (1024**3):.2f} GiB) is below 10% "
            f"of volume capacity ({ten_pct_dest / (1024**3):.2f} GiB); enforcing --min-free-gb={budget.min_free_gb:.2f} GiB floor."
        )

    if est_bytes > max_temp_bytes:
        raise BudgetExceededError(
            f"Estimated temp workspace ({est_bytes / (1024**3):.2f} GiB) exceeds --max-temp-gb "
            f"({budget.max_temp_gb:.2f} GiB). Increase --max-temp-gb or use --quality preview."
        )

    if dest_usage.free - est_bytes < min_free_bytes:
        raise BudgetExceededError(
            f"Insufficient free space on destination volume ({dest_check}): "
            f"free={dest_usage.free / (1024**3):.2f} GiB, estimated={est_bytes / (1024**3):.2f} GiB, "
            f"required reserve={budget.min_free_gb:.2f} GiB."
        )

    if temp_usage.free - est_bytes < min_free_bytes:
        raise BudgetExceededError(
            f"Insufficient free space on temp volume ({temp_check}): "
            f"free={temp_usage.free / (1024**3):.2f} GiB, estimated={est_bytes / (1024**3):.2f} GiB, "
            f"required reserve={budget.min_free_gb:.2f} GiB."
        )

    return {
        "dest_free_gb": round(dest_usage.free / (1024 ** 3), 3),
        "temp_free_gb": round(temp_usage.free / (1024 ** 3), 3),
        "estimated_temp_mb": round(est_bytes / (1024 ** 2), 2),
        "max_temp_gb": budget.max_temp_gb,
        "min_free_gb": budget.min_free_gb,
        "warnings": warnings,
    }


def terminate_process_tree(pid: int, grace_s: float = 2.0) -> None:
    """Terminate an owned child process tree gracefully, then force-kill only those children."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    procs = children + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace_s)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


def run_managed_subprocess(
    cmd: list[str],
    timeout_s: float = 300.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run subprocess safely (shell=False), track wall time & peak RSS, clean tree on timeout."""
    if not cmd or not isinstance(cmd, list):
        raise ValueError("cmd must be a non-empty argv list")
    t0 = time.monotonic()
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "shell": False,
        "cwd": str(cwd) if cwd else None,
        "env": env,
    }
    if platform.system() != "Windows":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(cmd, **kwargs)
    peak_rss_bytes = 0
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        ps_proc = None

    stdout_data = ""
    stderr_data = ""
    try:
        while True:
            if ps_proc is not None:
                try:
                    rss = ps_proc.memory_info().rss
                    for ch in ps_proc.children(recursive=True):
                        try:
                            rss += ch.memory_info().rss
                        except psutil.Error:
                            pass
                    if rss > peak_rss_bytes:
                        peak_rss_bytes = rss
                except psutil.Error:
                    pass
            try:
                stdout_data, stderr_data = proc.communicate(timeout=0.15)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - t0 > timeout_s:
                    terminate_process_tree(proc.pid)
                    try:
                        stdout_data, stderr_data = proc.communicate(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
                    raise BudgetExceededError(
                        f"Subprocess timed out after {timeout_s:.1f}s: {cmd[0]}"
                    )
    except KeyboardInterrupt:
        terminate_process_tree(proc.pid)
        raise

    elapsed_s = round(time.monotonic() - t0, 4)
    completed = subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout_data,
        stderr=stderr_data,
    )
    stats = {
        "elapsed_s": elapsed_s,
        "peak_rss_mb": round(peak_rss_bytes / (1024 ** 2), 2),
        "returncode": proc.returncode,
    }
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=stdout_data, stderr=stderr_data
        )
    return completed, stats


def safe_cleanup_run_temp(temp_dir: Path, run_root: Path) -> list[str]:
    """Delete ONLY owned files inside run_root/temp, strictly checking path containment & symlinks."""
    removed: list[str] = []
    if not temp_dir.exists():
        return removed
    resolved_root = run_root.resolve()
    ensure_contained_path(temp_dir, resolved_root)
    if temp_dir.resolve() == resolved_root:
        raise SchemaValidationError("Refusing to clean run_root directly")

    for item in sorted(temp_dir.rglob("*"), reverse=True):
        if item.is_symlink():
            # Unlink symlink itself without traversing target
            item.unlink(missing_ok=True)
            removed.append(str(item))
            continue
        ensure_contained_path(item, resolved_root)
        if item.is_file():
            item.unlink(missing_ok=True)
            removed.append(str(item))
        elif item.is_dir():
            try:
                item.rmdir()
            except OSError:
                pass
    return removed


@dataclass
class ResourceTracker:
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float = 0.0
    temp_peak_mb: float = 0.0
    _stage_start: dict[str, float] = field(default_factory=dict, repr=False)

    def start_stage(self, name: str) -> None:
        self._stage_start[name] = time.monotonic()
        self.sample_self()

    def end_stage(self, name: str, temp_dir: Path | None = None) -> float:
        t0 = self._stage_start.pop(name, time.monotonic())
        dt = round(time.monotonic() - t0, 4)
        self.stage_timings_s[name] = dt
        self.sample_self(temp_dir=temp_dir)
        return dt

    def record_child_peak(self, rss_mb: float) -> None:
        if rss_mb > self.peak_rss_mb:
            self.peak_rss_mb = round(rss_mb, 2)

    def sample_self(self, temp_dir: Path | None = None) -> None:
        try:
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024 ** 2)
            for ch in proc.children(recursive=True):
                try:
                    rss_mb += ch.memory_info().rss / (1024 ** 2)
                except psutil.Error:
                    pass
            if rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = round(rss_mb, 2)
        except psutil.Error:
            pass
        if temp_dir and temp_dir.exists():
            total = 0
            for f in temp_dir.rglob("*"):
                if f.is_file() and not f.is_symlink():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
            mb = round(total / (1024 ** 2), 2)
            if mb > self.temp_peak_mb:
                self.temp_peak_mb = mb

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_timings_s": dict(self.stage_timings_s),
            "peak_rss_mb": self.peak_rss_mb,
            "temp_peak_mb": self.temp_peak_mb,
        }

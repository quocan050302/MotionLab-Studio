"""Phase F — Async orchestration, bounded resource governance, disk safety, and process cleanup."""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

import psutil

from motionlab.config import ResourceConfig, ensure_safe_subpath

T = TypeVar("T")

GIB = 1024 ** 3
MIB = 1024 ** 2


class ResourceBudgetError(RuntimeError):
    """Raised when preflight or runtime resource/disk/timeout bounds are violated."""


@dataclass
class ResourceSnapshot:
    logical_cpus: int
    total_ram_gb: float
    available_ram_gb: float
    dest_total_gb: float
    dest_free_gb: float
    temp_total_gb: float
    temp_free_gb: float
    reserve_gb: float
    estimated_temp_gb: float
    analysis_workers: int
    render_workers: int
    ffmpeg_threads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_cpus": self.logical_cpus,
            "total_ram_gb": round(self.total_ram_gb, 3),
            "available_ram_gb": round(self.available_ram_gb, 3),
            "dest_total_gb": round(self.dest_total_gb, 3),
            "dest_free_gb": round(self.dest_free_gb, 3),
            "temp_total_gb": round(self.temp_total_gb, 3),
            "temp_free_gb": round(self.temp_free_gb, 3),
            "reserve_gb": round(self.reserve_gb, 3),
            "estimated_temp_gb": round(self.estimated_temp_gb, 4),
            "analysis_workers": self.analysis_workers,
            "render_workers": self.render_workers,
            "ffmpeg_threads": self.ffmpeg_threads,
        }


def estimate_intermediate_bytes(
    duration_s: float,
    width: int,
    height: int,
    fps: float,
    quality: str = "final",
    overlay_events: int = 0,
) -> int:
    """Conservatively estimate temp/output disk usage without full-video PNG dumps."""
    # Compressed H.264 intermediate + final output + small contact sheet/evidence + bounded overlays
    pixels = max(1, width * height)
    bpp_factor = 0.14 if quality == "final" else 0.08
    estimated_bps = max(1_500_000.0, pixels * max(fps, 1.0) * bpp_factor)
    video_bytes = int((estimated_bps / 8.0) * max(duration_s, 1.0))
    # Account for partial output + segment/overlay overhead + audio analysis PCM chunk (16kHz mono 16-bit)
    audio_pcm_bytes = int(16000 * 2 * max(duration_s, 1.0))
    overlay_bytes = int(overlay_events * 8 * MIB)
    overhead = 64 * MIB
    return int((video_bytes * 2.5) + audio_pcm_bytes + overlay_bytes + overhead)


@dataclass
class ResourceBudget:
    """Computes and enforces CPU, RAM, disk, and subprocess concurrency limits."""

    config: ResourceConfig
    dest_dir: Path
    temp_dir: Path
    snapshot: ResourceSnapshot | None = None
    peak_rss_mb: float = 0.0
    peak_temp_bytes: int = 0
    _active_procs: set[subprocess.Popen[Any]] = field(default_factory=set, repr=False)

    def preflight_budget(
        self,
        duration_s: float,
        width: int,
        height: int,
        fps: float,
        quality: str = "final",
        overlay_events: int = 0,
    ) -> ResourceSnapshot:
        """Measure actual CPU/RAM/disk and fail closed with actionable message if reserve violated."""
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        cpus = os.cpu_count() or 2
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / GIB
        avail_ram_gb = vm.available / GIB

        dest_usage = shutil.disk_usage(self.dest_dir)
        temp_usage = shutil.disk_usage(self.temp_dir)
        dest_total_gb = dest_usage.total / GIB
        dest_free_gb = dest_usage.free / GIB
        temp_total_gb = temp_usage.total / GIB
        temp_free_gb = temp_usage.free / GIB

        # Suggested reserve: max(5 GiB, 10% volume capacity) unless overridden by --min-free-gb
        if self.config.min_free_gb is not None:
            reserve_gb = self.config.min_free_gb
        else:
            min_vol_gb = min(dest_total_gb, temp_total_gb)
            reserve_gb = max(5.0, min(min_vol_gb * 0.10, 20.0))

        est_bytes = estimate_intermediate_bytes(
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
            quality=quality,
            overlay_events=overlay_events,
        )
        est_gb = est_bytes / GIB

        # Compute bounded worker counts: <= min(4, cores//2, memory-based capacity), floor 1
        mem_cap_analysis = max(1, int(avail_ram_gb // 0.8))
        default_analysis = max(1, min(4, max(1, cpus // 2), mem_cap_analysis))
        analysis_workers = (
            max(1, min(self.config.max_workers, default_analysis))
            if self.config.max_workers is not None
            else default_analysis
        )

        # Render workers: 1 by default, up to 2 only when plenty of free RAM (>= 4 GiB) & requested
        mem_cap_render = 2 if avail_ram_gb >= 4.0 and cpus >= 6 else 1
        render_workers = max(1, min(self.config.max_render_workers, mem_cap_render))
        if self.config.max_workers is not None:
            render_workers = max(1, min(render_workers, self.config.max_workers))

        # Avoid multiplying FFmpeg threads by external workers beyond logical core budget
        if self.config.ffmpeg_threads_per_worker is not None:
            ffmpeg_threads = max(1, min(self.config.ffmpeg_threads_per_worker, max(1, cpus // render_workers)))
        else:
            ffmpeg_threads = max(1, min(4, max(1, cpus // max(2, render_workers * 2))))

        snap = ResourceSnapshot(
            logical_cpus=cpus,
            total_ram_gb=total_ram_gb,
            available_ram_gb=avail_ram_gb,
            dest_total_gb=dest_total_gb,
            dest_free_gb=dest_free_gb,
            temp_total_gb=temp_total_gb,
            temp_free_gb=temp_free_gb,
            reserve_gb=reserve_gb,
            estimated_temp_gb=est_gb,
            analysis_workers=analysis_workers,
            render_workers=render_workers,
            ffmpeg_threads=ffmpeg_threads,
        )
        self.snapshot = snap

        if est_gb > self.config.max_temp_gb:
            raise ResourceBudgetError(
                f"Estimated intermediate size ({est_gb:.3f} GiB) exceeds --max-temp-gb ({self.config.max_temp_gb:.2f} GiB). "
                f"Increase --max-temp-gb or use --quality preview."
            )

        if temp_free_gb - est_gb < reserve_gb:
            raise ResourceBudgetError(
                f"Insufficient temp disk reserve on {self.temp_dir}: free={temp_free_gb:.2f} GiB, "
                f"estimated_need={est_gb:.3f} GiB, required_reserve={reserve_gb:.2f} GiB. "
                f"Free disk space or adjust --min-free-gb."
            )

        if dest_free_gb - est_gb < reserve_gb:
            raise ResourceBudgetError(
                f"Insufficient destination disk reserve on {self.dest_dir}: free={dest_free_gb:.2f} GiB, "
                f"estimated_need={est_gb:.3f} GiB, required_reserve={reserve_gb:.2f} GiB. "
                f"Free disk space or adjust --min-free-gb."
            )

        self.sample_metrics()
        return snap

    def check_runtime_disk_guard(self) -> None:
        """Check actual owned temp bytes and live free space; abort cleanly if reserve violated."""
        self.sample_metrics()
        if self.snapshot is None:
            return
        max_bytes = int(self.config.max_temp_gb * GIB)
        if self.peak_temp_bytes > max_bytes:
            raise ResourceBudgetError(
                f"Runtime temp usage ({self.peak_temp_bytes / GIB:.3f} GiB) exceeded max_temp_gb ({self.config.max_temp_gb:.2f} GiB)."
            )
        temp_free_gb = shutil.disk_usage(self.temp_dir).free / GIB
        if temp_free_gb < self.snapshot.reserve_gb:
            raise ResourceBudgetError(
                f"Runtime disk reserve violated on {self.temp_dir}: free={temp_free_gb:.2f} GiB < reserve={self.snapshot.reserve_gb:.2f} GiB."
            )

    def sample_metrics(self) -> None:
        """Record peak RSS (process + owned children) and peak owned temp bytes."""
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            rss_mb = rss / MIB
            if rss_mb > self.peak_rss_mb:
                self.peak_rss_mb = rss_mb
        except Exception:
            pass

        if self.temp_dir.exists():
            total = 0
            try:
                for p in self.temp_dir.rglob("*"):
                    if p.is_file():
                        try:
                            total += p.stat().st_size
                        except OSError:
                            pass
            except OSError:
                pass
            if total > self.peak_temp_bytes:
                self.peak_temp_bytes = total

    def cleanup_owned_temp(self, run_dir: Path, keep_on_fail: bool = False) -> None:
        """Remove ONLY run-owned temp/ directory after canonical containment check (never touches source/other projects)."""
        if keep_on_fail:
            return
        if not self.temp_dir.exists():
            return
        safe_temp = ensure_safe_subpath(run_dir, self.temp_dir)
        if safe_temp == run_dir.resolve():
            raise ResourceBudgetError("Refusing to delete run_dir root as temp.")
        shutil.rmtree(safe_temp, ignore_errors=True)

    def terminate_process_tree(self, proc: subprocess.Popen[Any], grace_s: float = 2.0) -> None:
        """Gracefully terminate ONLY an owned child process tree, then force-kill if stubborn."""
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                parent.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            _, alive = psutil.wait_procs([parent, *children], timeout=grace_s)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            try:
                proc.kill()
            except OSError:
                pass


def run_subprocess_guarded(
    cmd: list[str],
    budget: ResourceBudget | None = None,
    timeout_s: int | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess safely (shell=False, argv list, timeout, tree cleanup, metrics sampling)."""
    effective_timeout = timeout_s or (budget.config.subprocess_timeout_s if budget else 300)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if budget:
        budget._active_procs.add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=effective_timeout)
        if budget:
            budget.sample_metrics()
        res = subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr=stderr)
        if check and res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, cmd, output=stdout, stderr=stderr)
        return res
    except subprocess.TimeoutExpired as exc:
        if budget:
            budget.terminate_process_tree(proc)
        else:
            proc.kill()
        raise ResourceBudgetError(f"Subprocess timed out after {effective_timeout}s: {' '.join(cmd[:6])}") from exc
    finally:
        if budget:
            budget._active_procs.discard(proc)


async def run_bounded_async(
    tasks: list[tuple[str, Callable[[], T]]],
    max_concurrency: int,
) -> dict[str, T]:
    """Execute independent blocking/IO tasks asynchronously under an explicit semaphore, preserving deterministic key order."""
    sem = asyncio.Semaphore(max(1, max_concurrency))
    results: dict[str, T] = {}

    async def _worker(name: str, fn: Callable[[], T]) -> tuple[str, T]:
        async with sem:
            value = await asyncio.to_thread(fn)
            return name, value

    coros: list[Coroutine[Any, Any, tuple[str, T]]] = [_worker(name, fn) for name, fn in tasks]
    gathered = await asyncio.gather(*coros)
    # Single parent collects in deterministic original order
    by_name = dict(gathered)
    for name, _ in tasks:
        results[name] = by_name[name]
    return results

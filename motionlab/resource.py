"""Phase F — Async orchestration, bounded resource governance, disk safety, and process cleanup."""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

import psutil

from motionlab.config import ResourceConfig, ensure_safe_subpath

T = TypeVar("T")

GIB = 1024 ** 3
MIB = 1024 ** 2
GLOBAL_JOB_SEMAPHORE = threading.BoundedSemaphore(2)


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

    @property
    def analysis_workers(self) -> int:
        if self.snapshot is not None:
            return int(self.snapshot.analysis_workers)
        return int(self.config.max_workers or 2)

    @property
    def render_workers(self) -> int:
        if self.snapshot is not None:
            return int(self.snapshot.render_workers)
        return int(self.config.max_render_workers or 1)

    @property
    def ffmpeg_threads(self) -> int:
        if self.snapshot is not None:
            return int(self.snapshot.ffmpeg_threads)
        return int(self.config.ffmpeg_threads_per_worker or 2)

    @property
    def min_free_gb(self) -> float:
        if self.snapshot is not None:
            return float(self.snapshot.reserve_gb)
        return float(self.config.min_free_gb or 5.0)

    @property
    def max_temp_gb(self) -> float:
        return float(self.config.max_temp_gb)

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


def measure_process_tree_rss_mb(root_pid: int | None = None) -> float:
    """Measure RSS (MiB) of root_pid plus ALL recursive descendant child processes."""
    pid = root_pid if root_pid is not None else os.getpid()
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return round(total / MIB, 2)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def _dir_size_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


class ResourceTracker:
    """Tracks per-stage wall clock time, full process-tree peak RSS, and temp disk footprint."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.stages: dict[str, dict[str, Any]] = {}
        self.peak_tree_rss_mb: float = measure_process_tree_rss_mb()
        self.peak_temp_bytes: int = 0

    def start_stage(self, name: str) -> None:
        self._starts[name] = time.perf_counter()
        self.peak_tree_rss_mb = max(self.peak_tree_rss_mb, measure_process_tree_rss_mb())

    def end_stage(self, name: str, temp_dir: Path | None = None) -> dict[str, Any]:
        t0 = self._starts.pop(name, time.perf_counter())
        elapsed = round(time.perf_counter() - t0, 4)
        rss = measure_process_tree_rss_mb()
        tbytes = _dir_size_bytes(temp_dir)
        self.peak_tree_rss_mb = max(self.peak_tree_rss_mb, rss)
        self.peak_temp_bytes = max(self.peak_temp_bytes, tbytes)
        rec = {
            "elapsed_s": elapsed,
            "tree_rss_mb": rss,
            "temp_mb": round(tbytes / MIB, 2),
        }
        self.stages[name] = rec
        return rec

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_tree_rss_mb": round(self.peak_tree_rss_mb, 2),
            "peak_temp_mb": round(self.peak_temp_bytes / MIB, 2),
            "stages": dict(self.stages),
        }


class RuntimeResourceWatchdog:
    """Background watchdog monitoring owned child process trees, timeout, and disk safety."""

    def __init__(
        self,
        pids: list[int],
        timeout_s: float = 300.0,
        temp_dir: Path | None = None,
        min_free_gb: float = 2.0,
        max_temp_gb: float = 4.0,
        poll_interval_s: float = 0.25,
    ) -> None:
        self.pids = list(pids)
        self.timeout_s = float(timeout_s)
        self.temp_dir = temp_dir
        self.min_free_gb = float(min_free_gb)
        self.max_temp_gb = float(max_temp_gb)
        self.poll_interval_s = float(poll_interval_s)
        self.peak_rss_mb: float = 0.0
        self.peak_temp_bytes: int = 0
        self.violation: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_tree_rss_mb(self) -> float:
        return self.peak_rss_mb

    def _kill_pids(self) -> None:
        for pid in self.pids:
            try:
                p = psutil.Process(pid)
                for ch in p.children(recursive=True):
                    try:
                        ch.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            rss = measure_process_tree_rss_mb(os.getpid())
            for pid in self.pids:
                rss = max(rss, measure_process_tree_rss_mb(pid))
            self.peak_rss_mb = max(self.peak_rss_mb, rss)

            if time.perf_counter() - t0 > self.timeout_s:
                self.violation = f"timeout_exceeded:{self.timeout_s}s"
                self._kill_pids()
                return

            if self.temp_dir and self.temp_dir.exists():
                tbytes = _dir_size_bytes(self.temp_dir)
                self.peak_temp_bytes = max(self.peak_temp_bytes, tbytes)
                if tbytes > int(self.max_temp_gb * GIB):
                    self.violation = f"max_temp_gb_exceeded:{tbytes / GIB:.2f}>{self.max_temp_gb}"
                    self._kill_pids()
                    return
                try:
                    free_gb = shutil.disk_usage(self.temp_dir).free / GIB
                    if free_gb < self.min_free_gb:
                        self.violation = f"min_free_gb_violated:{free_gb:.2f}<{self.min_free_gb}"
                        self._kill_pids()
                        return
                except OSError:
                    pass
            self._stop.wait(self.poll_interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def run_managed_subprocess(
    cmd: list[str],
    timeout_s: float = 300.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run subprocess safely (shell=False) using temp-file stdio (prevents Windows grandchild pipe-EOF hangs)."""
    import tempfile

    t0 = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+b") as out_f, tempfile.TemporaryFile(mode="w+b") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=out_f,
            stderr=err_f,
            shell=False,
        )
        wd = RuntimeResourceWatchdog(pids=[proc.pid], timeout_s=timeout_s)
        wd.start()
        try:
            proc.wait(timeout=max(1.0, timeout_s + 5.0))
        except subprocess.TimeoutExpired as exc:
            wd._kill_pids()
            wd.stop()
            raise ResourceBudgetError(f"Subprocess timed out after {timeout_s}s: {' '.join(cmd[:6])}") from exc
        finally:
            # Clean up any lingering owned child processes (e.g. headless Chrome workers) without touching unrelated apps
            try:
                p_self = psutil.Process(proc.pid)
                for ch in p_self.children(recursive=True):
                    try:
                        ch.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            wd.stop()

        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", errors="replace")
        stderr = err_f.read().decode("utf-8", errors="replace")

    elapsed = round(time.perf_counter() - t0, 4)
    cp = subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr=stderr)
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cmd, output=stdout, stderr=stderr)
    return cp, {"elapsed_s": elapsed, "peak_tree_rss_mb": round(wd.peak_rss_mb, 2)}

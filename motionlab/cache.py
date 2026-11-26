"""Phase B8: Content-addressed local analysis cache with locking, integrity verification, and safe pruning."""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import cv2

from motionlab import ANALYZER_VERSION
from motionlab.config import MotionLabConfig, REPO_ROOT
from motionlab.schemas import (
    AnalysisResult,
    MediaInfo,
    atomic_write_json,
    canonical_hash,
    ensure_contained_path,
)

DEFAULT_CACHE_DIR = REPO_ROOT / ".motionlab_cache" / "analysis"


class AnalysisCache:
    """Content-addressed, integrity-checked, lock-safe local cache."""

    def __init__(self, cache_dir: Path | None = None, max_entries: int = 128) -> None:
        self.cache_dir = (cache_dir or DEFAULT_CACHE_DIR).resolve()
        self.max_entries = max(4, max_entries)

    def make_key(
        self,
        media: MediaInfo,
        config: MotionLabConfig,
        ffmpeg_version: str = "unknown",
    ) -> str:
        key_payload = {
            "source_hash": media.input_hash,
            "analyzer_version": ANALYZER_VERSION,
            "opencv_version": cv2.__version__,
            "ffmpeg_version": ffmpeg_version,
            "analysis_fps": config.analysis_fps,
            "enable_face_tracking": config.enable_face_tracking,
            "audio_track": config.audio_track,
        }
        return canonical_hash(key_payload)

    def _entry_path(self, key: str) -> Path:
        p = self.cache_dir / f"{key}.json"
        return ensure_contained_path(p, self.cache_dir)

    @contextlib.contextmanager
    def _key_lock(self, key: str, timeout_s: float = 10.0) -> Iterator[None]:
        """Cross-platform per-key lock file using O_CREAT | O_EXCL."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = ensure_contained_path(self.cache_dir / f"{key}.lock", self.cache_dir)
        deadline = time.monotonic() + timeout_s
        fd: int | None = None
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode("ascii"))
                break
            except FileExistsError:
                # Reap stale lock older than 120s
                try:
                    if time.time() - lock_path.stat().st_mtime > 120.0:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            lock_path.unlink(missing_ok=True)

    def get(self, key: str) -> AnalysisResult | None:
        path = self._entry_path(key)
        if not path.is_file():
            return None
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            payload = wrapper.get("payload")
            expected_digest = wrapper.get("payload_sha256")
            if not isinstance(payload, dict) or not expected_digest:
                return None
            if canonical_hash(payload) != expected_digest:
                # Corrupted cache entry — invalidate cleanly
                path.unlink(missing_ok=True)
                return None
            if payload.get("analyzer_version") != ANALYZER_VERSION:
                return None
            return AnalysisResult.from_dict(payload)
        except Exception:
            return None

    def put(self, key: str, result: AnalysisResult) -> Path:
        payload = result.to_dict()
        digest = canonical_hash(payload)
        wrapper = {
            "cache_key": key,
            "stored_at_epoch": int(time.time()),
            "payload_sha256": digest,
            "payload": payload,
        }
        with self._key_lock(key):
            path = self._entry_path(key)
            atomic_write_json(path, wrapper, indent=2)
            self._enforce_bounds()
            return path

    def _enforce_bounds(self) -> None:
        if not self.cache_dir.is_dir():
            return
        entries = sorted(
            (p for p in self.cache_dir.glob("*.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in entries[self.max_entries :]:
            ensure_contained_path(stale, self.cache_dir)
            stale.unlink(missing_ok=True)

    def prune(self) -> int:
        """Explicitly prune all entries inside this cache directory only."""
        if not self.cache_dir.is_dir():
            return 0
        removed = 0
        for p in list(self.cache_dir.glob("*.json")) + list(self.cache_dir.glob("*.lock")):
            ensure_contained_path(p, self.cache_dir)
            if p.is_file():
                p.unlink(missing_ok=True)
                removed += 1
        return removed

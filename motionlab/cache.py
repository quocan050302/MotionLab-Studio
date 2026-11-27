"""Phase B8 — Content-addressed local cache with atomic writes, per-key locks, and integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from motionlab import ANALYZER_VERSION
from motionlab.config import atomic_write_json, canonical_json_dumps, ensure_safe_subpath


class LocalAnalysisCache:
    """Bounded, content-addressed cache keyed by source SHA256 + config hash + versions + audio stream."""

    def __init__(self, cache_dir: Path, max_entries: int = 200) -> None:
        self.cache_dir = cache_dir.resolve()
        self.max_entries = max(10, max_entries)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_key(
        source_hash: str,
        analysis_config_hash: str,
        audio_track: int | None,
        dependency_signature: str,
        analyzer_version: str = ANALYZER_VERSION,
    ) -> str:
        raw = json.dumps(
            {
                "source_hash": source_hash,
                "analysis_config_hash": analysis_config_hash,
                "audio_track": audio_track,
                "dependency_signature": dependency_signature,
                "analyzer_version": analyzer_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _entry_path(self, key: str) -> Path:
        if not key.isalnum() or len(key) > 64:
            raise ValueError(f"Invalid cache key: {key!r}")
        return ensure_safe_subpath(self.cache_dir, self.cache_dir / f"analysis_{key}.json")

    def _lock_path(self, key: str) -> Path:
        return ensure_safe_subpath(self.cache_dir, self.cache_dir / f"analysis_{key}.lock")

    def _acquire_lock(self, key: str, timeout_s: float = 15.0) -> Path:
        lock_path = self._lock_path(key)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(f"{os.getpid()}:{time.time():.3f}\n")
                return lock_path
            except FileExistsError:
                # Reclaim stale lock older than 120s
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > 120.0:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for cache lock {lock_path.name}")
                time.sleep(0.05)

    @staticmethod
    def _release_lock(lock_path: Path) -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def get(self, key: str) -> dict[str, Any] | None:
        entry_path = self._entry_path(key)
        if not entry_path.exists():
            return None
        try:
            wrapper = json.loads(entry_path.read_text(encoding="utf-8"))
            payload = wrapper["payload"]
            expected_sha = wrapper["payload_sha256"]
            actual_sha = hashlib.sha256(
                canonical_json_dumps(payload, indent=None).encode("utf-8")
            ).hexdigest()
            if actual_sha != expected_sha:
                # Corrupted cache entry — invalidate rather than returning stale/corrupt data
                entry_path.unlink(missing_ok=True)
                return None
            return payload
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            try:
                entry_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        entry_path = self._entry_path(key)
        lock_path = self._acquire_lock(key)
        try:
            payload_sha = hashlib.sha256(
                canonical_json_dumps(payload, indent=None).encode("utf-8")
            ).hexdigest()
            wrapper = {
                "key": key,
                "saved_at_epoch": round(time.time(), 3),
                "payload_sha256": payload_sha,
                "payload": payload,
            }
            atomic_write_json(entry_path, wrapper)
            self._prune_if_needed()
            return entry_path
        finally:
            self._release_lock(lock_path)

    def _prune_if_needed(self) -> None:
        entries = sorted(
            self.cache_dir.glob("analysis_*.json"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        for stale in entries[self.max_entries :]:
            try:
                ensure_safe_subpath(self.cache_dir, stale).unlink(missing_ok=True)
            except OSError:
                pass

    def prune_all(self) -> int:
        removed = 0
        for p in list(self.cache_dir.glob("analysis_*.json")) + list(self.cache_dir.glob("analysis_*.lock")):
            safe = ensure_safe_subpath(self.cache_dir, p)
            try:
                safe.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        return removed

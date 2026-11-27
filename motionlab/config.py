"""Configuration, constants, and path-safety helpers for MotionLab AI ML-01."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT

VALID_PROFILES = (
    "auto",
    "neutral",
    "talking-head",
    "product",
    "storytelling",
    "gameplay",
    "tutorial",
    "cinematic",
)

CONCRETE_PROFILES = tuple(p for p in VALID_PROFILES if p != "auto")

VALID_QUALITIES = ("preview", "final")
VALID_ASPECTS = ("source", "9:16", "16:9", "1:1")
VALID_ENGINES = ("auto", "ffmpeg", "remotion", "hyperframes")
VALID_COMPOSITIONS = ("legacy", "r3", "cinematic", "generative")
VALID_AUDIO_MODES = ("preserve-exact", "mix")
VALID_SEMANTIC_MODES = ("auto", "off")
VALID_SFX_MODES = ("auto", "off")
VALID_MUSIC_MODES = ("off", "local")


@dataclass(frozen=True)
class AnalysisConfig:
    """Deterministic settings controlling local video/audio signal analysis."""

    sample_fps: float = 6.0
    downscale_width: int = 160
    min_shot_interval_s: float = 1.0
    cut_threshold_sigma: float = 3.2
    cut_min_hist_diff: float = 0.28
    audio_sample_rate: int = 16000
    audio_hop_ms: int = 50
    enable_subject_tracking: bool = True
    subject_sample_fps: float = 3.0
    max_candidates_per_minute: int = 40
    use_cache: bool = True

    def validate(self) -> None:
        if not (1.0 <= self.sample_fps <= 30.0) or not math.isfinite(self.sample_fps):
            raise ValueError(f"sample_fps must be finite in [1.0, 30.0], got {self.sample_fps}")
        if not (64 <= self.downscale_width <= 640):
            raise ValueError(f"downscale_width must be in [64, 640], got {self.downscale_width}")
        if not (0.2 <= self.min_shot_interval_s <= 10.0):
            raise ValueError("min_shot_interval_s out of range")
        if self.audio_sample_rate not in (8000, 16000, 22050, 44100):
            raise ValueError(f"Unsupported audio_sample_rate: {self.audio_sample_rate}")
        if not (10 <= self.audio_hop_ms <= 250):
            raise ValueError("audio_hop_ms must be in [10, 250]")

    def canonical_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ResourceConfig:
    """Resource governance bounds for CPU, RAM, disk, and process timeouts."""

    max_workers: int | None = None
    max_render_workers: int = 1
    max_temp_gb: float = 4.0
    min_free_gb: float | None = None  # Defaults dynamically to max(5.0 GiB, 10% volume)
    ffmpeg_threads_per_worker: int | None = None
    subprocess_timeout_s: int = 600
    allow_low_disk_preview_fallback: bool = False

    def validate(self) -> None:
        if self.max_workers is not None and (self.max_workers < 1 or self.max_workers > 32):
            raise ValueError(f"max_workers must be in [1, 32], got {self.max_workers}")
        if self.max_render_workers < 1 or self.max_render_workers > 4:
            raise ValueError(f"max_render_workers must be in [1, 4], got {self.max_render_workers}")
        if not math.isfinite(self.max_temp_gb) or self.max_temp_gb <= 0.1:
            raise ValueError(f"max_temp_gb must be > 0.1, got {self.max_temp_gb}")
        if self.min_free_gb is not None and (not math.isfinite(self.min_free_gb) or self.min_free_gb < 0.5):
            raise ValueError(f"min_free_gb must be >= 0.5, got {self.min_free_gb}")
        if self.subprocess_timeout_s < 5 or self.subprocess_timeout_s > 7200:
            raise ValueError("subprocess_timeout_s must be in [5, 7200]")


@dataclass(frozen=True)
class EditConfig:
    """Top-level configuration for a single MotionLab ML-01 pipeline invocation."""

    profile: str = "auto"
    seed: int = 42
    aspect: str = "source"
    quality: str = "final"
    offline: bool = True
    strict: bool = False
    dry_run: bool = False
    overwrite: bool = False
    audio_track: int | None = None  # None = preserve all compatible source audio tracks
    preferred_engine: str = "auto"
    allow_remotion: bool = True
    allow_hyperframes: bool = True
    allow_hdr_tonemap: bool = False
    aspect_bg_mode: str = "blur"  # blur | solid | smart_crop
    density_scale: float = 1.0
    intensity_scale: float = 1.0
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    cache_dir: Path | None = None
    render_preview_9x16: bool = False
    # Reconciled R3/R4/R5 advanced composition, semantic, and audio contracts (default keeps legacy ML-01 behavior)
    composition_mode: str = "legacy"  # legacy | r3 | cinematic | generative
    semantic_mode: str = "auto"  # auto | off
    audio_mode: str = "preserve-exact"  # preserve-exact | mix
    sfx_mode: str = "auto"  # auto | off
    music_mode: str = "off"  # off | local
    strict_cinematic: bool = False
    style: str = "auto"
    motion_density: str = "adaptive"
    remotion_opted_in: bool = False

    @property
    def cache_enabled(self) -> bool:
        return bool(self.analysis.use_cache)

    @property
    def timeout_s(self) -> int:
        return int(self.resources.subprocess_timeout_s)

    @property
    def enable_remotion(self) -> bool:
        return bool(self.allow_remotion)

    def validate(self) -> None:
        if self.profile not in VALID_PROFILES:
            raise ValueError(f"Unknown profile '{self.profile}'. Valid: {', '.join(VALID_PROFILES)}")
        if self.aspect not in VALID_ASPECTS:
            raise ValueError(f"Unknown aspect '{self.aspect}'. Valid: {', '.join(VALID_ASPECTS)}")
        if self.quality not in VALID_QUALITIES:
            raise ValueError(f"Unknown quality '{self.quality}'. Valid: {', '.join(VALID_QUALITIES)}")
        if self.preferred_engine not in VALID_ENGINES:
            raise ValueError(f"Unknown engine '{self.preferred_engine}'. Valid: {', '.join(VALID_ENGINES)}")
        if self.composition_mode not in VALID_COMPOSITIONS:
            raise ValueError(f"Unknown composition_mode '{self.composition_mode}'. Valid: {', '.join(VALID_COMPOSITIONS)}")
        if self.audio_mode not in VALID_AUDIO_MODES:
            raise ValueError(f"Unknown audio_mode '{self.audio_mode}'. Valid: {', '.join(VALID_AUDIO_MODES)}")
        if self.semantic_mode not in VALID_SEMANTIC_MODES:
            raise ValueError(f"Unknown semantic_mode '{self.semantic_mode}'. Valid: {', '.join(VALID_SEMANTIC_MODES)}")
        if self.sfx_mode not in VALID_SFX_MODES:
            raise ValueError(f"Unknown sfx_mode '{self.sfx_mode}'. Valid: {', '.join(VALID_SFX_MODES)}")
        if self.music_mode not in VALID_MUSIC_MODES:
            raise ValueError(f"Unknown music_mode '{self.music_mode}'. Valid: {', '.join(VALID_MUSIC_MODES)}")
        if not self.offline:
            raise ValueError("Cloud/online editing mode is forbidden in ML-01; --offline must remain true.")
        if self.aspect_bg_mode not in ("blur", "solid", "smart_crop"):
            raise ValueError(f"Unknown aspect_bg_mode '{self.aspect_bg_mode}'")
        if not math.isfinite(self.density_scale) or not (0.0 <= self.density_scale <= 2.0):
            raise ValueError("density_scale must be finite in [0.0, 2.0]")
        if not math.isfinite(self.intensity_scale) or not (0.0 <= self.intensity_scale <= 1.5):
            raise ValueError("intensity_scale must be finite in [0.0, 1.5]")
        if self.audio_track is not None and self.audio_track < 0:
            raise ValueError(f"audio_track must be >= 0, got {self.audio_track}")
        self.analysis.validate()
        self.resources.validate()

    def canonical_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "profile": self.profile,
            "seed": self.seed,
            "aspect": self.aspect,
            "quality": self.quality,
            "offline": self.offline,
            "strict": self.strict,
            "audio_track": self.audio_track,
            "preferred_engine": self.preferred_engine,
            "allow_remotion": self.allow_remotion,
            "allow_hyperframes": self.allow_hyperframes,
            "allow_hdr_tonemap": self.allow_hdr_tonemap,
            "aspect_bg_mode": self.aspect_bg_mode,
            "density_scale": self.density_scale,
            "intensity_scale": self.intensity_scale,
            "analysis": asdict(self.analysis),
        }
        if self.composition_mode != "legacy" or self.audio_mode != "preserve-exact":
            d.update({
                "composition_mode": self.composition_mode,
                "semantic_mode": self.semantic_mode,
                "audio_mode": self.audio_mode,
                "sfx_mode": self.sfx_mode,
                "music_mode": self.music_mode,
                "style": self.style,
                "motion_density": self.motion_density,
            })
        return d

    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def ensure_safe_subpath(root: Path, candidate: Path) -> Path:
    """Resolve `candidate` and verify it stays strictly within `root` (anti-traversal, no symlink escape)."""
    root_resolved = root.resolve()
    cand_resolved = candidate.resolve()
    try:
        cand_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path traversal rejected: {candidate} escapes root {root_resolved}") from exc
    return cand_resolved


def validate_no_path_traversal(raw_path: str) -> None:
    """Reject obvious traversal patterns or null bytes in user/JSON string paths."""
    if "\x00" in raw_path:
        raise ValueError("Null byte in path is forbidden")
    parts = Path(raw_path).parts
    if ".." in parts:
        raise ValueError(f"Relative parent traversal ('..') is forbidden in path: {raw_path}")


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write UTF-8 text to disk via sibling temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def canonical_json_dumps(data: Any, indent: int | None = 2) -> str:
    """Serialize JSON canonically, strictly rejecting NaN / Infinity."""
    if indent is None:
        return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return json.dumps(data, sort_keys=True, allow_nan=False, indent=indent) + "\n"


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write canonical JSON with strict NaN/Inf rejection."""
    atomic_write_text(path, canonical_json_dumps(data, indent=2))


# Backward-compatible aliases for reconciled R2-R4 modules
MotionLabConfig = EditConfig


def find_local_bin(project_dir: Path, name: str) -> Path | None:
    """Locate a project-local npm CLI binary (Windows .cmd/.exe or POSIX) without global npx."""
    bin_dir = project_dir.resolve() / "node_modules" / ".bin"
    candidates = [f"{name}.cmd", f"{name}.exe", name] if os.name == "nt" else [name]
    for cand_name in candidates:
        cand = bin_dir / cand_name
        if cand.is_file():
            return cand
    return None


def find_browser_executable() -> str | None:
    """Locate an already-installed local Chrome/Edge/Chromium executable."""
    from motionlab.renderers.base import find_local_browser

    return find_local_browser()

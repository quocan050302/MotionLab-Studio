"""Runtime configuration, deterministic hashing, and tool discovery (doctor)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from motionlab import __version__
from motionlab.schemas import SchemaValidationError, canonical_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = Path(__file__).resolve().parent / "presets"
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
VALID_ASPECTS = ("source", "9:16")
VALID_QUALITIES = ("final", "preview")
VALID_ENGINES = ("auto", "ffmpeg", "hyperframes", "remotion")


@dataclass
class MotionLabConfig:
    profile: str = "auto"
    aspect: str = "source"
    quality: str = "final"
    seed: int = 42
    offline: bool = True
    strict: bool = False
    dry_run: bool = False
    overwrite: bool = False
    audio_track: int | None = None
    max_workers: int | None = None
    max_temp_gb: float = 4.0
    min_free_gb: float = 5.0
    cache_enabled: bool = True
    cache_dir: str | None = None
    enable_face_tracking: bool = True
    enable_hyperframes: bool = True
    enable_remotion: bool = False  # License-gated: must opt in after confirming Remotion license terms
    preferred_engine: str = "auto"
    allow_hdr_tonemap: bool = False
    allow_vfr_cfr_normalize: bool = True
    pad_background: str = "blur"  # blur | solid
    analysis_fps: float = 6.0
    render_preview_9x16: bool = False
    timeout_s: int = 600
    allow_split_layout: bool = False

    def validate(self) -> None:
        if self.profile not in VALID_PROFILES:
            raise SchemaValidationError(f"Unknown profile: {self.profile!r}. Valid: {VALID_PROFILES}")
        if self.aspect not in VALID_ASPECTS:
            raise SchemaValidationError(f"Unsupported aspect: {self.aspect!r}. Valid: {VALID_ASPECTS}")
        if self.quality not in VALID_QUALITIES:
            raise SchemaValidationError(f"Unsupported quality: {self.quality!r}. Valid: {VALID_QUALITIES}")
        if self.preferred_engine not in VALID_ENGINES:
            raise SchemaValidationError(
                f"Unsupported engine: {self.preferred_engine!r}. Valid: {VALID_ENGINES}"
            )
        if not self.offline:
            raise SchemaValidationError("Online/cloud mode is strictly forbidden in ML-01 (--offline must remain true)")
        if self.max_temp_gb <= 0.1:
            raise SchemaValidationError(f"max_temp_gb must be > 0.1, got {self.max_temp_gb}")
        if self.min_free_gb < 0.0:
            raise SchemaValidationError(f"min_free_gb must be >= 0, got {self.min_free_gb}")
        if self.analysis_fps <= 0.5 or self.analysis_fps > 30.0:
            raise SchemaValidationError(f"analysis_fps out of safe range (0.5..30): {self.analysis_fps}")
        if self.audio_track is not None and self.audio_track < 0:
            raise SchemaValidationError(f"audio_track must be >= 0, got {self.audio_track}")
        if self.pad_background not in ("blur", "solid"):
            raise SchemaValidationError(f"pad_background must be 'blur' or 'solid', got {self.pad_background}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        d = self.to_dict()
        # Exclude ephemeral output/execution flags that do not alter plan semantics
        for k in ("overwrite", "dry_run", "cache_dir", "cache_enabled", "max_workers", "max_temp_gb", "min_free_gb", "timeout_s"):
            d.pop(k, None)
        return canonical_hash(d)


def _run_version_probe(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        if proc.returncode == 0:
            out = (proc.stdout or proc.stderr or "").strip()
            return out.splitlines()[0] if out else "ok"
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def find_browser_executable() -> str | None:
    """Locate a local Chrome/Chromium executable without downloading anything."""
    env_override = os.environ.get("MOTIONLAB_BROWSER_EXECUTABLE")
    if env_override and Path(env_override).is_file():
        return str(Path(env_override).resolve())

    candidates: list[Path] = []
    sys_name = platform.system()
    if sys_name == "Darwin":
        candidates.extend([
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ])
    elif sys_name == "Windows":
        for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
            if root:
                candidates.append(Path(root) / "Google/Chrome/Application/chrome.exe")
                candidates.append(Path(root) / "Microsoft/Edge/Application/msedge.exe")
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found:
                return found

    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def find_local_bin(renderer_dir: Path, bin_name: str) -> Path | None:
    """Locate a project-local npm binary (Windows .cmd or POSIX executable)."""
    bin_dir = renderer_dir / "node_modules" / ".bin"
    if platform.system() == "Windows":
        for ext in (".cmd", ".exe", ""):
            p = bin_dir / f"{bin_name}{ext}"
            if p.is_file():
                return p
    else:
        p = bin_dir / bin_name
        if p.is_file():
            return p
    return None


def detect_ffmpeg_capabilities(ffmpeg_bin: str) -> dict[str, Any]:
    """Query actual FFmpeg encoders and filters via subprocess."""
    caps: dict[str, Any] = {"encoders": [], "filters_checked": {}}
    try:
        enc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=8, shell=False,
        )
        wanted_enc = ("libx264", "h264_videotoolbox", "h264_nvenc", "h264_qsv", "libvpx-vp9", "aac", "png")
        found_enc = [e for e in wanted_enc if e in enc.stdout]
        caps["encoders"] = found_enc
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        flt = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=8, shell=False,
        )
        wanted_flt = ("scale", "crop", "overlay", "gblur", "eq", "curves", "vignette", "blend", "alphamerge", "zscale")
        caps["filters_checked"] = {f: (f" {f} " in flt.stdout) for f in wanted_flt}
    except (OSError, subprocess.SubprocessError):
        pass
    return caps


def run_doctor(config: MotionLabConfig | None = None) -> dict[str, Any]:
    """Inspect local environment, binaries, renderers, licenses, and disk capacity."""
    cfg = config or MotionLabConfig()
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    node_bin = shutil.which("node")
    npm_bin = shutil.which("npm")
    browser_bin = find_browser_executable()

    ffmpeg_ver = _run_version_probe([ffmpeg_bin, "-version"]) if ffmpeg_bin else None
    ffprobe_ver = _run_version_probe([ffprobe_bin, "-version"]) if ffprobe_bin else None
    node_ver = _run_version_probe([node_bin, "--version"]) if node_bin else None
    npm_ver = _run_version_probe([npm_bin, "--version"]) if npm_bin else None
    ffmpeg_caps = detect_ffmpeg_capabilities(ffmpeg_bin) if ffmpeg_bin else {"encoders": [], "filters_checked": {}}

    # Check Python libraries
    py_mods: dict[str, str | None] = {}
    for mod_name in ("numpy", "PIL", "cv2", "psutil"):
        try:
            mod = __import__(mod_name)
            py_mods[mod_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            py_mods[mod_name] = None

    # Check bundled OpenCV Haar cascades
    haar_available = False
    try:
        import cv2  # type: ignore

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        haar_available = cascade_path.is_file()
    except Exception:
        haar_available = False

    # Check HyperFrames local project
    hf_dir = REPO_ROOT / "renderers" / "hyperframes"
    hf_bin = find_local_bin(hf_dir, "hyperframes")
    hf_available = bool(hf_bin and node_bin and browser_bin and ffmpeg_bin)

    # Check Remotion local project (license-gated!)
    rem_dir = REPO_ROOT / "renderers" / "remotion"
    rem_bin = find_local_bin(rem_dir, "remotion")
    rem_installed = bool(rem_bin and node_bin and browser_bin)
    rem_available = bool(rem_installed and cfg.enable_remotion)

    # Check Manim optional
    manim_bin = shutil.which("manim")

    # Disk space on repo root
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = round(usage.free / (1024 ** 3), 2)
    total_gb = round(usage.total / (1024 ** 3), 2)

    # Check if cloud API keys exist in env (NEVER print values!)
    cloud_env_keys_present = [
        k for k in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
        if bool(os.environ.get(k))
    ]

    status = "OK" if (ffmpeg_bin and ffprobe_bin and all(py_mods.values())) else "MISSING_REQUIRED"

    return {
        "motionlab_version": __version__,
        "status": status,
        "offline_enforced": True,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count() or 1,
        },
        "disk": {
            "path": str(REPO_ROOT),
            "total_gb": total_gb,
            "free_gb": free_gb,
            "below_10pct_warning": usage.free < int(usage.total * 0.10),
        },
        "tools": {
            "ffmpeg": {"path": ffmpeg_bin, "version": ffmpeg_ver, "capabilities": ffmpeg_caps},
            "ffprobe": {"path": ffprobe_bin, "version": ffprobe_ver},
            "node": {"path": node_bin, "version": node_ver},
            "npm": {"path": npm_bin, "version": npm_ver},
            "browser": {"path": browser_bin, "available": bool(browser_bin)},
        },
        "python_dependencies": py_mods,
        "local_models": {
            "opencv_haar_face": {
                "available": haar_available,
                "license": "Apache-2.0 / Intel OpenCV bundled (no runtime download)",
            }
        },
        "engines": {
            "ffmpeg": {
                "available": bool(ffmpeg_bin and ffprobe_bin and py_mods.get("cv2") and py_mods.get("numpy")),
                "license": "External FFmpeg binary (GPL/LGPL depending on build); Python compositor MIT/Apache/BSD",
                "role": "primary",
            },
            "hyperframes": {
                "installed": bool(hf_bin),
                "available": hf_available and cfg.enable_hyperframes,
                "binary": str(hf_bin) if hf_bin else None,
                "license": "Apache-2.0 (hyperframes) + Standard No-Charge GSAP License",
                "role": "secondary_overlays",
            },
            "remotion": {
                "installed": rem_installed,
                "license_opt_in": cfg.enable_remotion,
                "available": rem_available,
                "binary": str(rem_bin) if rem_bin else None,
                "license": "Remotion Source-Available License (free <=3-person org/individual; paid >=4-person company)",
                "role": "secondary_overlays_license_gated",
            },
            "manim": {
                "installed": bool(manim_bin),
                "available": False,
                "reason": "Optional out-of-critical-path engine (requires explicit template/data; not enabled at ML-01)",
                "license": "MIT",
            },
        },
        "security_audit": {
            "cloud_api_calls_enabled": False,
            "cloud_env_keys_detected_count": len(cloud_env_keys_present),
            "note": "API keys are never read, printed, or transmitted by motionlab runtime.",
        },
    }

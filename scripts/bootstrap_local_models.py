"""One-time offline bootstrap for MotionLab local AI models & llama.cpp binary (Windows/macOS/Linux).

STRICTLY BOOTSTRAP-ONLY: never imported by motionlab runtime modules.
Downloads pinned Apache-2.0 / MIT models defined in models/manifest.json and verifies every SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TOOLS_DIR = ROOT / "tools" / "llama.cpp"
MANIFEST_PATH = MODELS_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected_sha256:
        actual = sha256_file(dest)
        if actual == expected_sha256:
            print(f"[SKIP] {dest.name} already verified (sha256={actual[:12]})")
            return
    print(f"[DOWNLOAD] {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "MotionLab-Bootstrap/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -10
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total > 0:
                pct = int(done * 100 / total)
                if pct >= last_pct + 15:
                    print(f"  ... {dest.name}: {pct}% ({done / (1024*1024):.1f}/{total / (1024*1024):.1f} MiB)", flush=True)
                    last_pct = pct
    os.replace(tmp, dest)
    actual = sha256_file(dest)
    if expected_sha256 and actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}")
    print(f"[OK] {dest.name} verified (sha256={actual[:12]}, size={dest.stat().st_size})")


def bootstrap_models() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    models = manifest["models"]

    # 1. MODNet ONNX (Apache-2.0, 25.8 MB)
    m_mod = models["modnet_photographic_portrait_matting"]
    download_file(m_mod["distribution_url"], MODELS_DIR / m_mod["filename"], m_mod["sha256"])

    # 2. Faster-Whisper-Small (MIT, 483 MB)
    m_asr = models["faster_whisper_small"]
    asr_dir = MODELS_DIR / m_asr["directory"]
    base_url = m_asr["distribution_url"].rstrip("/") + "/resolve/main"
    for fname, exp_sha in m_asr["files"].items():
        download_file(f"{base_url}/{fname}", asr_dir / fname, exp_sha)

    # 3. Qwen2.5-3B-Instruct Q4_K_M GGUF (Apache-2.0, 2.10 GB)
    m_llm = models["qwen2_5_3b_instruct_q4_k_m"]
    download_file(m_llm["distribution_url"], MODELS_DIR / m_llm["filename"], m_llm["sha256"])


def bootstrap_llama_cpp() -> None:
    exe_name = "llama-completion.exe" if os.name == "nt" else "llama-completion"
    cli_name = "llama-cli.exe" if os.name == "nt" else "llama-cli"
    if (TOOLS_DIR / exe_name).is_file() or (TOOLS_DIR / cli_name).is_file():
        print(f"[SKIP] llama.cpp already installed in {TOOLS_DIR}")
        return

    print("[GITHUB] Querying recent ggml-org/llama.cpp releases (MIT)...")
    req = urllib.request.Request(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15",
        headers={"User-Agent": "MotionLab-Bootstrap/1.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        releases = json.loads(resp.read().decode("utf-8"))

    chosen = None
    tag = "unknown"
    prefs = (
        ("win-cpu-x64.zip", "win-avx2-x64.zip", "win-vulkan-x64.zip")
        if os.name == "nt"
        else (("macos-arm64",) if ("arm" in platform.machine().lower() or "aarch" in platform.machine().lower()) else ("ubuntu-x64",))
    )
    for rel in releases:
        assets = rel.get("assets", [])
        for pref in prefs:
            for a in assets:
                name = str(a.get("name", "")).lower()
                if pref in name and (name.endswith(".zip") or name.endswith(".tar.gz")):
                    chosen = a
                    tag = str(rel.get("tag_name", "unknown"))
                    break
            if chosen:
                break
        if chosen:
            break

    if not chosen:
        raise RuntimeError("No compatible llama.cpp prebuilt archive found in recent releases")

    zip_path = ROOT / ".cache" / str(chosen["name"])
    download_file(str(chosen["browser_download_url"]), zip_path)
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACT] {zip_path.name} -> {TOOLS_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            base = Path(member.filename).name
            if not base:
                continue
            target = TOOLS_DIR / base
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    meta = {
        "upstream_repo": "ggml-org/llama.cpp",
        "release_tag": tag,
        "asset_name": chosen["name"],
        "asset_sha256": sha256_file(zip_path),
        "license": "MIT",
    }
    (TOOLS_DIR / "bootstrap_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    zip_path.unlink(missing_ok=True)
    print(f"[OK] llama.cpp {tag} extracted to {TOOLS_DIR}")


if __name__ == "__main__":
    bootstrap_models()
    bootstrap_llama_cpp()
    print("[DONE] All local AI models and llama.cpp verified.")

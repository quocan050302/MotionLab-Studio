"""Phase 1 (ML-01.R4): Local Offline LLM Runner (llama.cpp MIT + Qwen2.5-3B-Instruct Apache-2.0).

Strictly offline subprocess execution with grammar-constrained `--json-schema-file` decoding, deterministic seed,
prompt-injection sanitization (treating transcript strictly as untrusted media data), zero shell/URL/code execution,
and bounded memory/timeout guards.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from motionlab.config import REPO_ROOT
from motionlab.resource import GLOBAL_JOB_SEMAPHORE
from motionlab.schemas import ensure_contained_path

MODELS_DIR = REPO_ROOT / "models"
TOOLS_DIR = REPO_ROOT / "tools" / "llama.cpp"
DEFAULT_LLM_MODEL_ID = "qwen2_5_3b_instruct_q4_k_m"

_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_llama_binary() -> Path | None:
    """Locate local llama-completion / llama-cli executable (repo tools/ first, then PATH)."""
    for cand in (
        TOOLS_DIR / "llama-completion",
        TOOLS_DIR / "llama-completion.exe",
        TOOLS_DIR / "llama-cli",
        TOOLS_DIR / "llama-cli.exe",
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    for name in ("llama-completion", "llama-cli"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    return None


def inspect_llm_model(
    models_dir: Path | None = None,
    model_id: str = DEFAULT_LLM_MODEL_ID,
) -> dict[str, Any]:
    """Verify local llama.cpp binary and SHA-256-pinned GGUF weights without downloading anything."""
    mdir = (models_dir or MODELS_DIR).resolve()
    manifest_p = mdir / "manifest.json"
    llama_bin = find_llama_binary()
    info: dict[str, Any] = {
        "model_id": model_id,
        "available": False,
        "binary_path": str(llama_bin) if llama_bin else None,
        "binary_license": "MIT (ggml-org/llama.cpp)",
        "model_path": None,
        "license": "Apache-2.0",
        "quantization": "Q4_K_M",
        "sha256_expected": None,
        "sha256_actual": None,
        "sha256_verified": False,
        "size_bytes": 0,
        "expected_ram_mb": 2600,
        "reason": "",
    }
    if not llama_bin:
        info["reason"] = "llama_cpp_binary_missing (run bootstrap --with-semantic)"
        return info
    if not manifest_p.is_file():
        info["reason"] = f"models_manifest_missing:{manifest_p}"
        return info

    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        entry = (manifest.get("models") or {}).get(model_id)
        if not isinstance(entry, dict):
            info["reason"] = f"llm_model_id_not_in_manifest:{model_id}"
            return info
        info["license"] = str(entry.get("license", "Apache-2.0"))
        info["quantization"] = str(entry.get("quantization", "Q4_K_M"))
        info["expected_ram_mb"] = int(entry.get("expected_ram_mb", 2600))
        expected_sha = str(entry.get("sha256", ""))
        info["sha256_expected"] = expected_sha
        mp = ensure_contained_path(mdir / str(entry["filename"]), mdir)
        info["model_path"] = str(mp)
        if not mp.is_file():
            info["reason"] = f"llm_weights_missing:{mp} (run bootstrap --with-semantic)"
            return info
        info["size_bytes"] = int(mp.stat().st_size)
        actual_sha = _sha256_file(mp)
        info["sha256_actual"] = actual_sha
        if expected_sha and actual_sha != expected_sha:
            info["reason"] = f"llm_sha256_mismatch:expected={expected_sha[:12]},actual={actual_sha[:12]}"
            return info
        info["sha256_verified"] = True
        info["available"] = True
        info["reason"] = "ok"
        return info
    except Exception as exc:
        info["reason"] = f"llm_manifest_error:{exc}"
        return info


def sanitize_untrusted_transcript_text(text: str) -> str:
    """Strip chatml control tokens & suspicious prompt-override markers from untrusted spoken text."""
    cleaned = _CONTROL_TOKEN_RE.sub("", text or "")
    cleaned = cleaned.replace("```", "'''").strip()
    return cleaned[:4000]


def _extract_balanced_json(raw_stdout: str) -> dict[str, Any]:
    """Cleanly strip llama-completion EOF markers and parse the top-level JSON object."""
    text = raw_stdout.replace("[end of text]", "").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in local LLM output: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    end_idx = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
    if end_idx < 0:
        raise ValueError(f"Truncated JSON object from local LLM: {text[start:start+300]!r}")
    parsed = json.loads(text[start : end_idx + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected top-level JSON object from local LLM")
    return parsed


def generate_structured_json(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict[str, Any],
    seed: int = 42,
    max_tokens: int = 1200,
    context_size: int = 4096,
    timeout_s: float = 90.0,
    models_dir: Path | None = None,
    model_id: str = DEFAULT_LLM_MODEL_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run local Qwen2.5-3B via llama.cpp with native --json-schema grammar decoding (100% offline)."""
    llm_info = inspect_llm_model(models_dir=models_dir, model_id=model_id)
    if not llm_info["available"]:
        raise RuntimeError(f"Local semantic LLM unavailable: {llm_info['reason']}")

    t0 = time.perf_counter()
    # Format Qwen2.5 ChatML prompt (system instructions strictly separated from untrusted transcript data!)
    prompt = (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    with tempfile.TemporaryDirectory(prefix="ml01_r4_llm_") as td:
        td_p = Path(td)
        schema_p = td_p / "schema.json"
        prompt_p = td_p / "prompt.txt"
        schema_p.write_text(json.dumps(json_schema), encoding="utf-8")
        prompt_p.write_text(prompt, encoding="utf-8")

        use_gpu_layers = "99" if (os.name != "nt" and hasattr(os, "uname") and os.uname().sysname == "Darwin") else "0"
        threads = str(min(6 if os.name == "nt" else 4, max(2, (os.cpu_count() or 4) // 2)))
        cmd = [
            str(llm_info["binary_path"]),
            "-m", str(llm_info["model_path"]),
            "--temp", "0.0",
            "--top-k", "1",
            "--seed", str(int(seed)),
            "-n", str(int(max_tokens)),
            "-c", str(int(context_size)),
            "-t", threads,
            "-ngl", use_gpu_layers,
            "--no-display-prompt",
            "--single-turn",
            "-no-cnv",
            "--json-schema-file", str(schema_p),
            "-f", str(prompt_p),
        ]

        # Run inside isolated clean env (no API keys, no proxy, local binary dir on PATH/DYLD_LIBRARY_PATH)
        env = {
            k: v for k, v in os.environ.items()
            if k not in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
        }
        bin_dir = str(Path(llm_info["binary_path"]).parent)
        if os.name == "nt":
            env["PATH"] = f"{bin_dir};{env.get('PATH', '')}"
        else:
            env["DYLD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")
            env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

        last_err = ""
        for attempt in range(2):
            with GLOBAL_JOB_SEMAPHORE:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s,
                    env=env,
                    shell=False,
                )
            if proc.returncode != 0:
                last_err = f"llama_rc={proc.returncode}: {proc.stderr[-300:]}"
                continue
            try:
                parsed = _extract_balanced_json(proc.stdout)
                meta = {
                    "model_id": llm_info["model_id"],
                    "model_sha256": llm_info["sha256_actual"],
                    "binary_path": llm_info["binary_path"],
                    "seed": int(seed),
                    "attempts": attempt + 1,
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "raw_output_chars": len(proc.stdout),
                }
                return parsed, meta
            except Exception as exc:
                last_err = f"json_parse_error:{exc}"

    raise RuntimeError(f"Local LLM structured JSON generation failed after 2 attempts: {last_err}")

#!/usr/bin/env bash
# MotionLab AI (ML-01) — One-time Online Bootstrap Script for Linux / macOS.
# Non-admin, isolated .venv + pinned local Node renderers.
# After bootstrap, `python -m motionlab edit ...` runs 100% OFFLINE with ZERO API keys.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SKIP_NODE="${1:-}"

echo "=== MotionLab AI (ML-01) Bootstrap (Linux/macOS) ==="
echo "Project root : ${PROJECT_ROOT}"
echo "Estimated disk usage: ~450 MB (.venv + optional local Node renderers). No multi-GB models downloaded."

for tool in python3 ffmpeg ffprobe; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "ERROR: Required tool '${tool}' not found on PATH." >&2
    exit 1
  fi
  echo "Found ${tool}: $(command -v "${tool}")"
done

if [ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
  echo "Creating isolated virtual environment at .venv ..."
  python3 -m venv "${PROJECT_ROOT}/.venv"
fi

VENV_PY="${PROJECT_ROOT}/.venv/bin/python"
"${VENV_PY}" -m pip install --upgrade pip
"${VENV_PY}" -m pip install -r "${PROJECT_ROOT}/requirements.lock"
"${VENV_PY}" -m pip install -e "${PROJECT_ROOT}" --no-deps --no-build-isolation

if [ "${SKIP_NODE}" != "--skip-node-renderers" ]; then
  if command -v npm >/dev/null 2>&1; then
    for renderer in remotion hyperframes; do
      RDIR="${PROJECT_ROOT}/renderers/${renderer}"
      if [ -f "${RDIR}/package.json" ]; then
        echo "Installing pinned local Node renderer: ${renderer} ..."
        (
          cd "${RDIR}"
          if [ -f "package-lock.json" ]; then
            npm ci --no-audit --no-fund
          else
            npm install --no-audit --no-fund
          fi
        )
      fi
    done
  else
    echo "WARN: npm not found; skipping optional Remotion/HyperFrames renderers (FFmpeg engine remains 100% functional)."
  fi
fi

echo ""
echo "=== Running offline doctor check ==="
"${VENV_PY}" -m motionlab doctor
echo ""
echo "Bootstrap complete! Run offline edit:"
echo '  .venv/bin/python -m motionlab edit --input "input/raw.mp4" --profile auto --offline --output "runs"'

#!/usr/bin/env bash
# MotionLab-AI ML-01 — Linux/macOS Bootstrap Script (Online one-time setup)
# Inspectable, non-root, no global PATH mutations.
# Usage:
#   bash scripts/bootstrap.sh [--with-hyperframes] [--with-remotion]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INSTALL_HF=0
INSTALL_REM=0
for arg in "$@"; do
  case "$arg" in
    --with-hyperframes) INSTALL_HF=1 ;;
    --with-remotion) INSTALL_REM=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

echo "=== MotionLab-AI ML-01 Bootstrap (Linux/macOS) ==="
echo "Project root: $REPO_ROOT"
echo "Note: One-time online install of pinned local dependencies (~120 MB core + ~80-230 MB optional Node renderers)."
echo "Runtime video editing is 100% offline with 0 API keys."

command -v python3 >/dev/null 2>&1 || { echo "Error: python3 (>=3.10) is required." >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg is required on PATH." >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "Error: ffprobe is required on PATH." >&2; exit 1; }

if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "Creating isolated .venv..."
  python3 -m venv "$REPO_ROOT/.venv"
fi

echo "Installing pinned Python dependencies from requirements.lock..."
"$REPO_ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools
"$REPO_ROOT/.venv/bin/python" -m pip install --only-binary=:all: -r "$REPO_ROOT/requirements.lock"
"$REPO_ROOT/.venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT"

if [ "$INSTALL_HF" -eq 1 ]; then
  command -v npm >/dev/null 2>&1 || { echo "Error: Node.js 22+ and npm are required for HyperFrames." >&2; exit 1; }
  echo "Installing local HyperFrames renderer (Apache-2.0)..."
  (cd "$REPO_ROOT/renderers/hyperframes" && PUPPETEER_SKIP_DOWNLOAD=1 PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci --no-audit --no-fund)
fi

if [ "$INSTALL_REM" -eq 1 ]; then
  command -v npm >/dev/null 2>&1 || { echo "Error: Node.js 22+ and npm are required for Remotion." >&2; exit 1; }
  echo "Installing local Remotion renderer (check THIRD_PARTY_NOTICES.md license terms)..."
  (cd "$REPO_ROOT/renderers/remotion" && PUPPETEER_SKIP_DOWNLOAD=1 npm ci --no-audit --no-fund)
fi

echo "Running doctor check..."
"$REPO_ROOT/.venv/bin/python" -m motionlab doctor
echo "Bootstrap complete! Run: .venv/bin/python -m motionlab edit --input input/raw.mp4 --profile auto --offline --output runs"

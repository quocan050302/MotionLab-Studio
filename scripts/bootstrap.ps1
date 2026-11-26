# MotionLab-AI ML-01 — Windows PowerShell Bootstrap Script (Online one-time setup)
# Inspectable, non-admin, no global PATH mutations, no permanent ExecutionPolicy changes.
# Usage:
#   powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\bootstrap.ps1
# Optional flags:
#   -InstallHyperFrames    Install Apache-2.0 HyperFrames + GSAP local renderer (~80 MB)
#   -InstallRemotion       Install license-gated Remotion local renderer (~150 MB; see THIRD_PARTY_NOTICES.md)

[CmdletBinding()]
param(
    [switch]$InstallHyperFrames,
    [switch]$InstallRemotion
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

Write-Host "=== MotionLab-AI ML-01 Bootstrap (Windows PowerShell) ==="
Write-Host "Project root: $RepoRoot"
Write-Host "Note: This step installs pinned local dependencies online once. Runtime editing is 100% offline with 0 API keys."
Write-Host "Estimated disk space: ~120 MB (.venv Python core) + ~80-230 MB if Node renderers selected."

# 1. Verify Python >= 3.10 and FFmpeg/ffprobe
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) { throw "Python 3.10+ not found on PATH. Install Python 3.10-3.13 first." }
$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobeCmd = Get-Command ffprobe -ErrorAction SilentlyContinue
if (-not $ffmpegCmd -or -not $ffprobeCmd) {
    throw "ffmpeg and ffprobe are required on PATH. Install FFmpeg (e.g. winget install Gyan.FFmpeg) and re-run."
}

# 2. Create isolated .venv
$venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating isolated .venv..."
    & python -m venv (Join-Path $RepoRoot ".venv")
}

Write-Host "Installing pinned Python dependencies from requirements.lock..."
& $venvPy -m pip install --upgrade pip setuptools
& $venvPy -m pip install --only-binary=:all: -r (Join-Path $RepoRoot "requirements.lock")
& $venvPy -m pip install --no-deps --no-build-isolation -e $RepoRoot

# 3. Optional Node renderers (HyperFrames Apache-2.0 / Remotion license-gated)
if ($InstallHyperFrames) {
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) { throw "Node.js 22+ and npm are required for HyperFrames." }
    Write-Host "Installing local HyperFrames renderer (Apache-2.0)..."
    $env:PUPPETEER_SKIP_DOWNLOAD = "1"
    $env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = "1"
    Push-Location (Join-Path $RepoRoot "renderers\hyperframes")
    try { & npm ci --no-audit --no-fund } finally { Pop-Location }
}

if ($InstallRemotion) {
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) { throw "Node.js 22+ and npm are required for Remotion." }
    Write-Host "Installing local Remotion renderer (SEE THIRD_PARTY_NOTICES.md for license terms)..."
    $env:PUPPETEER_SKIP_DOWNLOAD = "1"
    Push-Location (Join-Path $RepoRoot "renderers\remotion")
    try { & npm ci --no-audit --no-fund } finally { Pop-Location }
}

Write-Host "Running doctor check..."
& $venvPy -m motionlab doctor
Write-Host "Bootstrap complete! Run: .\.venv\Scripts\python.exe -m motionlab edit --input input\raw.mp4 --profile auto --offline --output runs"

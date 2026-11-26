# MotionLab-AI ML-01 — Windows PowerShell Offline Smoke Verification Script
# Usage:
#   powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\smoke.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$venvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw ".venv not found. Run scripts\bootstrap.ps1 first." }

# Explicitly clear cloud API env vars in process scope only (never modifies user env)
Remove-Item Env:ELEVENLABS_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue

Write-Host "1/4 Doctor check..."
& $venvPy -m motionlab doctor

Write-Host "2/4 Effects catalog & gallery check..."
& $venvPy -m motionlab effects list
& $venvPy -m motionlab effects gallery --output (Join-Path $RepoRoot "runs\smoke_ps1")

Write-Host "3/4 Generating procedural 6s 9:16 fixture..."
$fixture = Join-Path $RepoRoot "runs\smoke_ps1\smoke_fixture_9x16.mp4"
& $venvPy -c "from pathlib import Path; from tests.fixtures.generator import generate_demo_9x16_15s; generate_demo_9x16_15s(Path(r'$fixture'), width=540, height=960, fps=30, duration_s=6.0)"

Write-Host "4/4 Running offline one-command edit..."
& $venvPy -m motionlab edit --input $fixture --profile auto --seed 42 --offline --no-cache --output (Join-Path $RepoRoot "runs\smoke_ps1")
Write-Host "Smoke test PASSED!"

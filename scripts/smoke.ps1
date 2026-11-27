<#
.SYNOPSIS
  Quick local offline smoke test for MotionLab AI ML-01.
  Strips API key environment variables, runs doctor, effects list, and verifies a procedural edit end-to-end.
#>
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$VenvPy = "$ProjectRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    throw "Virtual environment missing. Run .\scripts\bootstrap.ps1 first."
}

# Strip known cloud API keys from process scope
foreach ($k in @("ELEVENLABS_API_KEY","OPENAI_API_KEY","ANTHROPIC_API_KEY","GOOGLE_API_KEY","GEMINI_API_KEY")) {
    Remove-Item "Env:$k" -ErrorAction SilentlyContinue
}

& $VenvPy -m motionlab doctor
& $VenvPy -m motionlab effects list
& $VenvPy -m pytest -q tests/
Write-Host "Smoke check complete."

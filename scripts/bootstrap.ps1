<#
.SYNOPSIS
  MotionLab AI (ML-01) — One-time Online Bootstrap Script for Windows PowerShell.
  Inspectable, non-admin, pins dependencies in a local .venv + local Node renderer folders.
  After this script completes, `python -m motionlab edit ...` runs 100% OFFLINE with ZERO API keys.

.USAGE
  powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\bootstrap.ps1
  powershell -ExecutionPolicy Bypass -Scope Process -File .\scripts\bootstrap.ps1 -SkipNodeRenderers
#>
param(
    [switch]$SkipNodeRenderers
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "=== MotionLab AI (ML-01) Bootstrap (Windows PowerShell) ==="
Write-Host "Project root : $ProjectRoot"
Write-Host "Note         : This bootstrap uses internet ONCE to install pinned local dependencies."
Write-Host "               Estimated disk usage: ~450 MB (.venv + local Node renderers). No multi-GB models downloaded."
Write-Host "               Runtime editing (`python -m motionlab edit ...`) is 100% offline with ZERO API keys."

# 1. Verify required host binaries
foreach ($tool in @("python", "ffmpeg", "ffprobe")) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Required tool '$tool' not found on PATH. Install it first (no global PATH changes performed by this script)."
    }
    Write-Host "Found $tool : $($cmd.Source)"
}

# 2. Check free disk space (>= 2 GiB minimum for bootstrap)
$drive = (Get-Item $ProjectRoot).PSDrive
$freeGiB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "Drive $($drive.Name): free space = $freeGiB GiB"
if ($freeGiB -lt 2.0) {
    throw "Insufficient free disk space on $($drive.Name): ($freeGiB GiB < 2.0 GiB required for bootstrap)."
}

# 3. Create isolated Python virtual environment (.venv)
if (-not (Test-Path "$ProjectRoot\.venv\Scripts\python.exe")) {
    Write-Host "Creating isolated virtual environment at .venv ..."
    python -m venv "$ProjectRoot\.venv"
}
$VenvPy = "$ProjectRoot\.venv\Scripts\python.exe"

Write-Host "Installing pinned Python dependencies from requirements.lock ..."
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r "$ProjectRoot\requirements.lock"
& $VenvPy -m pip install -e "$ProjectRoot" --no-deps --no-build-isolation

# 4. Install optional local Node renderers (Remotion + HyperFrames) using committed package-lock.json
if (-not $SkipNodeRenderers) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if ($npm) {
        foreach ($renderer in @("remotion", "hyperframes")) {
            $rDir = Join-Path $ProjectRoot "renderers\$renderer"
            if (Test-Path (Join-Path $rDir "package.json")) {
                Write-Host "Installing pinned local Node renderer: $renderer ..."
                Push-Location $rDir
                try {
                    if (Test-Path "package-lock.json") {
                        npm ci --no-audit --no-fund
                    } else {
                        npm install --no-audit --no-fund
                    }
                } finally {
                    Pop-Location
                }
            }
        }
    } else {
        Write-Warning "npm not found; skipping optional Remotion/HyperFrames local renderers (FFmpeg engine remains 100% functional)."
    }
}

Write-Host ""
Write-Host "=== Running offline doctor check ==="
& $VenvPy -m motionlab doctor

Write-Host ""
Write-Host "Bootstrap complete! To edit a raw MP4 completely offline:"
Write-Host '  .\.venv\Scripts\python.exe -m motionlab edit --input "input\raw.mp4" --profile auto --offline --output "runs"'

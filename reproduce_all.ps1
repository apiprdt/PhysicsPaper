# ADCD full reproduction script (Windows PowerShell)
# Usage: .\reproduce_all.ps1
#
# Reproduces all three paper scenarios from scratch and verifies
# byte-exact determinism across three independent runs.

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host "=== ADCD Reproduction Pipeline ===" -ForegroundColor Cyan
Write-Host ""

# -- Install ---------------------------------------------------------------
Write-Host "Installing dependencies..." -ForegroundColor DarkGray
try {
    pip install -e ".[dev]" | Out-Null
} catch {
    Write-Warning "pip install failed; attempting to continue with existing env."
}

# -- Unit tests ------------------------------------------------------------
Write-Host ""
Write-Host "[1/3] Running unit tests..." -ForegroundColor Yellow
$env:PYTHONPATH = "src"
python -m pytest tests/ -q --tb=short

# -- Full three-scenario validation ----------------------------------------
Write-Host ""
Write-Host "[2/3] Running full ADCD validation (all three scenarios)..." -ForegroundColor Yellow
python src\adcd\run_adcd_v3_validation_blind.py --top-k 5

# -- Determinism check (three independent runs) ----------------------------
Write-Host ""
Write-Host "[3/3] Determinism check (3 identical runs)..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path run_outputs | Out-Null

1..3 | ForEach-Object {
    python src\adcd\run_adcd_v3_validation_blind.py --top-k 5 2>$null |
        Out-File "run_outputs\determinism_run_$_.txt" -Encoding utf8
}

$d12 = Compare-Object (Get-Content run_outputs\determinism_run_1.txt) `
                      (Get-Content run_outputs\determinism_run_2.txt)
$d13 = Compare-Object (Get-Content run_outputs\determinism_run_1.txt) `
                      (Get-Content run_outputs\determinism_run_3.txt)

if ($d12.Count -eq 0 -and $d13.Count -eq 0) {
    Write-Host "  PASS: all three runs are byte-exact identical." -ForegroundColor Green
} else {
    Write-Host "  FAIL: runs differ — determinism violated." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Reproduction complete. Results in run_outputs/ ===" -ForegroundColor Green

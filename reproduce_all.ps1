# ADCD full reproduction script (Windows PowerShell)
# Usage: .\reproduce_all.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== ADCD Reproduction Pipeline ===" -ForegroundColor Cyan

pip install -e ".[dev]" | Out-Null

Write-Host "`n[1/8] Unit tests..." -ForegroundColor Yellow
py -3.11 -m pytest tests/ -q --tb=short

Write-Host "`n[2/8] Standard benchmark..." -ForegroundColor Yellow
py -3.11 run_correction_discovery.py --proposer mock

Write-Host "`n[3/8] Real-world benchmark..." -ForegroundColor Yellow
py -3.11 run_real_data_benchmark.py

Write-Host "`n[4/8] PySR baselines..." -ForegroundColor Yellow
py -3.11 run_pysr_baseline.py --profile fast
py -3.11 run_pysr_baseline.py --profile fair

Write-Host "`n[5/8] Ablation + oracle..." -ForegroundColor Yellow
py -3.11 run_ablation.py
py -3.11 run_oracle_ablation.py

Write-Host "`n[6/8] Correction scaling..." -ForegroundColor Yellow
py -3.11 run_correction_scaling.py

Write-Host "`n[7/8] Reports + figures..." -ForegroundColor Yellow
py -3.11 scripts/generate_experiment_report.py
py -3.11 scripts/generate_efficiency_table.py
py -3.11 scripts/validate_results.py
py -3.11 generate_figures.py

Write-Host "`n[8/8] Mercury diagnostic..." -ForegroundColor Yellow
py -3.11 scratch/investigate_mercury.py

Write-Host "`n=== Reproduction complete ===" -ForegroundColor Green

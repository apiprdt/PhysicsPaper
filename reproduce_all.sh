#!/usr/bin/env bash
# ADCD full reproduction script (cross-platform)
# Usage: bash reproduce_all.sh
#
# Works on Linux, macOS, and Windows (Git Bash / WSL).
# For pure Windows PowerShell see reproduce_all.ps1.

set -euo pipefail
cd "$(dirname "$0")"

echo -e "\e[36m=== ADCD Reproduction Pipeline ===\e[0m"

pip install -e ".[dev]" >/dev/null 2>&1 || {
  echo "WARN: pip install -e '.[dev]' failed; attempting to continue anyway"
}

echo -e "\e[33m\n[1/8] Unit tests...\e[0m"
python -m pytest tests/ -q --tb=short

echo -e "\e[33m\n[2/8] Standard benchmark...\e[0m"
python run_correction_discovery.py --proposer mock

echo -e "\e[33m\n[3/8] Real-world benchmark...\e[0m"
python run_real_data_benchmark.py

echo -e "\e[33m\n[4/8] PySR baselines...\e[0m"
python run_pysr_baseline.py --profile fast
python run_pysr_baseline.py --profile fair

echo -e "\e[33m\n[5/8] Ablation + oracle...\e[0m"
python run_ablation.py
python run_oracle_ablation.py

echo -e "\e[33m\n[6/8] Correction scaling...\e[0m"
python run_correction_scaling.py

echo -e "\e[33m\n[7/8] Reports + figures...\e[0m"
python scripts/generate_experiment_report.py
python scripts/generate_efficiency_table.py
python scripts/validate_results.py
python generate_figures.py

echo -e "\e[33m\n[8/8] Mercury real-data smoke test...\e[0m"
python -m pytest tests/test_real_data.py -k mercury -q --tb=short

echo -e "\e[32m\n=== Reproduction complete ===\e[0m"

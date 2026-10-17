#!/usr/bin/env bash
# ADCD full reproduction script
# Usage: bash reproduce_all.sh
#
# Reproduces all three paper scenarios from scratch and verifies
# byte-exact determinism across three independent runs.
#
# Works on Linux, macOS, and Windows (Git Bash / WSL).
# For pure Windows PowerShell, see reproduce_all.ps1

set -euo pipefail
cd "$(dirname "$0")"

echo "=== ADCD Reproduction Pipeline ==="
echo ""

# -- Install (editable, with dev extras) -----------------------------------
echo "Installing dependencies..."
pip install -e ".[dev]" > /dev/null 2>&1 || {
  echo "WARN: pip install -e '.[dev]' failed; attempting to continue with existing env"
}

# -- Unit tests ------------------------------------------------------------
echo ""
echo "[1/3] Running unit tests..."
export PYTHONPATH=src
python -m pytest tests/ -q --tb=short

# -- Full three-scenario validation ----------------------------------------
echo ""
echo "[2/3] Running full ADCD validation (all three scenarios)..."
python src/adcd/run_adcd_v3_validation_blind.py --top-k 5

# -- Determinism check (three independent runs) ----------------------------
echo ""
echo "[3/3] Determinism check (3 identical runs)..."
mkdir -p run_outputs

for i in 1 2 3; do
  python src/adcd/run_adcd_v3_validation_blind.py --top-k 5 \
    > run_outputs/determinism_run_${i}.txt 2>/dev/null
done

if diff -q run_outputs/determinism_run_1.txt run_outputs/determinism_run_2.txt > /dev/null && \
   diff -q run_outputs/determinism_run_1.txt run_outputs/determinism_run_3.txt > /dev/null; then
  echo "  PASS: all three runs are byte-exact identical."
else
  echo "  FAIL: runs differ — determinism violated."
  diff run_outputs/determinism_run_1.txt run_outputs/determinism_run_2.txt | head -20 || true
  exit 1
fi

echo ""
echo "=== Reproduction complete. Results in run_outputs/ ==="

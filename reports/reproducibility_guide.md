# ADCD Reproducibility Guide

This guide details how to reproduce all benchmark tables, figures, and empirical evaluations on branch `Exploring`.

---

## 1. Environment Requirements

- **Operating System:** Windows, Linux, or macOS
- **Python Version:** Python 3.11+
- **Core Dependencies:**
  ```bash
  pip install numpy scipy sympy pytest flake8
  ```
- Optional (for JAX-accelerated Stage 1 pipeline):
  ```bash
  pip install jax jaxlib
  ```

---

## 2. Test Suite & Static Analysis

Run the full regression test suite (71 tests):
```bash
pytest tests/ -v
```
Run static analysis (must report zero warnings):
```bash
flake8 src/ tests/ scripts/ spike/
```

---

## 3. Reproducing Key Experiments

### A. Real-World SPARC RAR Empirical Audit (2,693 Points)
To download the official SPARC catalog (`data/RAR.mrt`) and execute the blind candidate evaluation:
```bash
python scripts/audit_sparc_rar.py
```
- **Output Report:** `reports/real_world_sparc_audit.md`
- **Output Data:** `reports/real_world_sparc_fits.csv`

### B. Sphere Drag Prediction Flip Benchmark
To execute the multi-seed calibration test demonstrating the transition from Equivalence Class ($\sigma = 0.3\%$) to Uniquely Identifiable ($\sigma = 0.1\%$):
```bash
python -c "from calibration.run import run_sphere_drag_flip_test; print(run_sphere_drag_flip_test())"
```
- **Output Data:** `reports/sphere_drag_flip_results.csv` and `reports/calibration_seed_log.csv`

### C. Oracle Leak Regression Suite
To verify that all evaluation engines are strictly blind to ground truth:
```bash
pytest tests/test_no_oracle_leak.py -v
```

### D. Multi-Seed Asimov Calibration Suite
To run the full multi-seed Asimov evaluation across Screened Coulomb, Entropy Expansion, and Time Dilation:
```bash
python -m calibration.run
```
- **Output Report:** `reports/calibration_report.md`
- **Output Log:** `reports/calibration_seed_log.csv`

---

## 4. Repository Structure Overview

- `src/adcd/`: Core production modules (frozen, blind likelihood, scale-normalized Hessian, identifiability protocol).
- `tests/`: Complete regression suite verifying numerical stability and zero oracle leakage.
- `scripts/`: Data fetching and domain-specific audit scripts (e.g. SPARC RAR).
- `reports/`: Markdown and CSV artifacts documenting all experimental runs.
- `spike/`: Isolated experimental research harnesses (e.g. `crossover.py`), strictly segregated from core.

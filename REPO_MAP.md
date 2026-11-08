# Repository Map — ADCD (PhysicsPaper)

This document provides a comprehensive structural guide to the codebase, entry points, execution commands for canonical scenarios, and environment requirements.

## 1. System Requirements & Environment
- **Python Version:** `>=3.10` (Primary pinned environment: Python 3.11.x, x86_64).
- **Core Dependencies (Pinned in `requirements-lock.txt`):**
  - `jax==0.10.1`, `jaxlib==0.10.1` (CPU float64 via `jax_enable_x64=True`)
  - `numpy==2.4.3`
  - `scipy==1.17.1`
  - `sympy==1.14.0`
  - `pytest==9.0.3`
  - `flake8==7.3.0`
  - `pint==0.23`

## 2. Directory Structure
```
PhysicsPaper/
├── .github/workflows/       # Continuous Integration workflows (ci.yml, publish.yml)
├── audit/                   # Audit scripts and validation patches
├── eval/                    # Metrics computation and evaluation helpers
├── notebooks/               # Interactive exploration & quickstart
│   └── 01_quickstart.ipynb  # End-to-end Screened Coulomb walkthrough
├── paper/                   # LaTeX source files for the NeurIPS paper
├── rebuttal/                # Rebuttal experiments, multi-seed scripts, data
│   ├── multi_seed_stability.py
│   └── results/multi_seed_stability.json
├── run_outputs/             # Artifacts, validation reports, determinism logs
├── src/adcd/                # Core library source code
│   ├── anomaly_scenarios.py             # Canonical physics scenarios & data generators
│   ├── arc_scorer.py                    # Asymptotic Residual Curvature regime scoring
│   ├── asymptotic_dictionary_proposer_v3.py # 8 regularized primitives & grammar proposer
│   ├── auto_scenario.py                 # Custom data loading & scenario inference
│   ├── bayesian_ranker.py               # Pareto ranking & model selection
│   ├── budget_sweep.py                  # Search space size vs recovery evaluation
│   ├── cli.py                           # CLI interface (`adcd` console script)
│   ├── coarse_evaluator.py              # Gate 2 screening
│   ├── constants.py                     # Physical constants and thresholds
│   ├── context.py                       # ProposalContext dataclasses
│   ├── dimensional_checker.py           # 5D dimensional analysis & AST validator
│   ├── grammar_proposer_v3.py           # AST generation engine
│   ├── identifiability.py               # Identifiability metric definitions
│   ├── jax_optimizer.py                 # L-BFGS-B nonlinear optimization via JAX
│   ├── jax_precision_config.py          # Global float64 precision setup
│   ├── metrics.py                       # NMSE, BIC, structure classification
│   ├── mode_detection.py                # Additive vs Multiplicative correction detector
│   ├── pipeline.py                      # Stage 1 candidate filtering pipeline
│   ├── quickfit.py                      # Closed-form linear sieve & taxonomy priors
│   ├── real_data_loader.py              # Data loading utilities
│   ├── real_scenarios.py                # SPARC & real astrophysics benchmark loaders
│   ├── residual_features.py             # Feature engineering on residuals
│   └── run_adcd_v3_validation_blind.py  # 4-step formal protocol runner
├── tests/                   # Pytest test suite
│   ├── test_auto_scenario.py
│   ├── test_placeholder.py
│   └── test_regression_v1.py
├── pyproject.toml           # Build configuration & packaging metadata
├── requirements-lock.txt    # Frozen exact dependencies
├── requirements.txt         # Runtime requirements
└── reproduce_all.ps1        # PowerShell reproduction script
```

## 3. Entry Points & How to Run

### Running Canonical Scenarios via Validation Protocol
The formal 4-step protocol (`budget_disclosure`, `positive_control`, `ablation_control`, `determinism_check`) is executed via:
```powershell
# Run all three canonical scenarios (Screened Coulomb, Entropy Expansion, Time Dilation)
python src/adcd/run_adcd_v3_validation_blind.py --top-k 5

# Run a single scenario:
python src/adcd/run_adcd_v3_validation_blind.py --scenario "Screened Coulomb"
python src/adcd/run_adcd_v3_validation_blind.py --scenario "Entropy Expansion"
python src/adcd/run_adcd_v3_validation_blind.py --scenario "Time Dilation"

# Run in pure blind mode (no domain taxonomy prior):
python src/adcd/run_adcd_v3_validation_blind.py --no-taxonomy
```

### Running the CLI
```powershell
adcd --version
adcd list-scenarios
adcd list-taxonomy
adcd fit --data <csv_file> --x-col <x> --y-col <y> --baseline-expr "<expr>"
```

### Running Tests and Lint
```powershell
# Unit test suite
pytest tests/ -q

# Code quality / linting
flake8 src tests
```

### Multi-Seed Stability Benchmark
```powershell
python rebuttal/multi_seed_stability.py
```
Outputs are saved to `rebuttal/results/multi_seed_stability.json`.

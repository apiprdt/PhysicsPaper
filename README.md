# ADCD — Anomaly-Driven Correction Discovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20534940.svg)](https://doi.org/10.5281/zenodo.20534940)
[![CI](https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml/badge.svg)](https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery**

ADCD is a symbolic regression framework that discovers *physical correction terms* rather than learning equations from scratch. Given a known classical law and anomalous observations, ADCD recovers the dimensionless correction Δ that reconciles theory with experiment — mirroring how physics actually evolves.

> **94.4% structural class match rate** across 9 complex physical anomalies under noise up to 10%.

---

## Key Features

- **Correction-first paradigm** — starts from a known classical law, not a blank slate
- **4-gate physics filter cascade** — AST complexity, dimensional homogeneity, transcendental argument constraints, and asymptotic consistency (ARC)
- **JAX-traced L-BFGS-B optimizer** — fast, differentiable parameter fitting
- **BIC reranking** — selects the most parsimonious correction over purely numerical fits
- **Noise-robust** — 100% accuracy at 0–1% noise, 88.9% at 5–10% noise

## Quick Start

### Installation

```bash
pip install adcd
```

Or install from source:

```bash
git clone https://github.com/apiprdt/PhysicsPaper.git
cd adcd
pip install -e ".[dev]"
```

### Usage

Running ADCD is now extremely simple using the high-level scientific API:

```python
import adcd

# 1. Load a pre-defined benchmark scenario
scenarios = adcd.get_all_scenarios()
scenario = scenarios[0]  # Relativistic Kinetic Energy

# 2. Run discovery in a single line!
result = adcd.discover_correction(scenario, max_iterations=5, proposer="mock")

print(f"Discovered correction: {result.best_expr}")
print(f"Residual NMSE: {result.best_nmse_residual:.2e}")
print(f"Parameters: {result.best_theta}")

# 3. Export LaTeX or plot residuals (great for Jupyter Notebooks)
print(result.export_latex())
# result.plot_residuals()
```

For custom experimental data, use `adcd.fit(...)`:

```python
import numpy as np
import adcd

# Generate some custom data (e.g., classical y = 2*x with a correction term)
x = np.linspace(1.0, 5.0, 100)
X = {"x": x}
y_classical = 2.0 * x
y_observed = 2.0 * x + 0.5 * x**2  # contains a hidden x**2 correction

# Discover the correction term
result = adcd.fit(
    X=X,
    y_obs=y_observed,
    y_classical=y_classical,
    limit_variable="x",
    limit_direction="0",
    correction_mode="additive"
)

result.summary()
# result.plot_residuals()
```

## Benchmark Results

| Scenario | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|----------|:--------:|:--------:|:--------:|:---------:|
| Relativistic KE | ✓ | ✓ | ✓ | ✓ |
| Yukawa Gravity | ✓ | ✓ | ✗ | ✗ |
| Anharmonic Spring | ✓ | ✓ | ✓ | ✓ |
| Screened Coulomb | ✓ | ✓ | ✓ | ✓ |
| Net Radiation | ✓ | ✓ | ✓ | ✓ |
| Stokes Drag | ✓ | ✓ | ✓ | ✓ |
| Maxwell-Boltzmann | ✓ | ✓ | ✓ | ✓ |
| Quantum Tunneling | ✓ | ✓ | ✓ | ✓ |
| Sinc Diffraction | ✓ | ✓ | ✓ | ✓ |
| **Overall** | **100%** | **100%** | **88.9%** | **88.9%** |

## Project Structure

```
adcd/
├── src/adcd/                  # Installable package
│   ├── __init__.py            # Public API
│   ├── anomaly_scenarios.py   # 9 benchmark scenarios
│   ├── arc_scorer.py          # Asymptotic consistency gate (ARC)
│   ├── coarse_evaluator.py    # Coarse numerical evaluation
│   ├── correction_orchestrator.py  # Main discovery loop
│   ├── dimensional_checker.py # Dimensional homogeneity gate
│   ├── jax_optimizer.py       # JAX L-BFGS-B optimizer
│   ├── llm_proposer.py        # Mock + LLM proposers
│   ├── metrics.py             # Evaluation metrics (NMSE, BIC, structural class)
│   ├── orchestrator.py        # Legacy orchestrator
│   ├── pipeline.py            # Stage 1 filter cascade
│   ├── real_data_loader.py    # Real-world data loading utilities
│   ├── real_scenarios.py      # Real-world validation scenarios
│   └── residual_analyzer.py   # Residual statistical features
├── tests/                     # Unit + integration tests
├── paper/                     # LaTeX source + figures
├── .github/workflows/         # CI + PyPI publish
├── pyproject.toml             # Build configuration
└── README.md                  # This file
```

## Running Tests

```bash
pytest --cov=adcd
```

## Reproducing Paper Results

```bash
python run_experiments.py          # Main 9-scenario benchmark
python run_ablation.py             # Ablation study
python run_pysr_baseline.py        # PySR comparison
python run_mlp_baseline.py         # MLP comparison
python run_reproducibility.py      # Full reproducibility suite
python generate_figures.py         # Generate all paper figures
```

## Citing This Work

If you use ADCD in your research, please cite:

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained
                Symbolic Regression for Evolutionary Scientific Discovery}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0.0},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```

## AI Disclosure

This project was developed with assistance from Google DeepMind's Antigravity AI assistant. AI was used as a pair-programming and writing tool. All scientific content, experimental design decisions, and intellectual contributions are the author's own.

## License

[MIT](LICENSE)

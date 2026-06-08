# ADCD — Anomaly-Driven Correction Discovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20534940.svg)](https://doi.org/10.5281/zenodo.20534940)
[![CI](https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml/badge.svg)](https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads//)

**Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery**

ADCD is a symbolic regression framework that discovers *physical correction terms* rather than learning equations from scratch. Given a known classical law and anomalous observations, ADCD recovers the dimensionless correction Δ that reconciles theory with experiment — mirroring how physics actually evolves.

> **94.4% structural class match rate** across 9 physical anomaly scenarios under noise up to 10%.  
> **58 automated unit tests** passing on Python 3.10 and 3.11.

---

## Key Features

- **Correction-first paradigm** — starts from a known classical law, not a blank slate
- **4-gate physics filter cascade** — AST complexity, dimensional homogeneity, transcendental argument constraints, and asymptotic consistency (ARC)
- **JAX-traced L-BFGS-B optimizer** — parameter-scaled differentiable fitting with multi-restart log-uniform initialization
- **BIC reranking** — selects the most parsimonious correction over purely numerical fits
- **Residual feature intelligence** — statistical priors bias the template sampler toward the correct mathematical family
- **Noise-robust** — 100% accuracy at 0–1% noise, 88.9% at 5–10% noise (deterministic seed)

## Quick Start

### Installation

```bash
pip install adcd
```

Or install from source:

```bash
git clone https://github.com/apiprdt/PhysicsPaper.git
cd PhysicsPaper
pip install -e ".[dev]"
```

### Usage

Running ADCD is extremely simple using the high-level scientific API:

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

# 3. Export LaTeX or plot residuals
print(result.export_latex())
result.plot_residuals()
```

For custom experimental data, use `adcd.fit(...)`:

```python
import numpy as np
import adcd

x = np.linspace(1.0, 5.0, 100)
X = {"x": x}
y_classical = 2.0 * x
y_observed  = 2.0 * x + 0.5 * x**2   # hidden x² correction

result = adcd.fit(
    X=X,
    y_obs=y_observed,
    y_classical=y_classical,
    limit_variable="x",
    limit_direction="0",
    correction_mode="additive"
)

result.summary()
```

## Benchmark Results

Results from `run_correction_discovery.py --proposer mock` (seed=42, 4 iterations per scenario).

| Scenario | Tier | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|----------|------|:--------:|:--------:|:--------:|:---------:|
| Relativistic KE | Textbook | ✓ | ✓ | ✓ | ✓ |
| Yukawa Gravity | Textbook | ✓ | ✓ | ✗ | ✗ |
| Anharmonic Spring | Textbook | ✓ | ✓ | ✓ | ✓ |
| Screened Coulomb | Cross-Domain | ✓ | ✓ | ✓ | ✓ |
| Net Radiation | Cross-Domain | ✓ | ✓ | ✓ | ✓ |
| Nonlinear Drag | Cross-Domain | ✓ | ✓ | ✓ | ✓ |
| Mystery-A (tanh²) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| Mystery-B (sinc) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| Mystery-C (log-quotient) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| **Overall** | | **100%** | **100%** | **88.9%** | **88.9%** |

> **Note**: Yukawa Gravity fails at ≥5% noise because exponential decay and logarithmic quotient are numerically indistinguishable at the tested SNR — an information-theoretic limit, not a framework deficiency.

### Real-World Physical Constants Benchmark

| Physical Scenario | Discovered Correction | Converged | Class Match |
|---|---|:---:|:---:|
| Mercury Perihelion (GR) | `θ₀ · GM/(c²r)` | ✓ | ✓ |
| Hydrogen Lamb Shift (QED) | `θ₀ · (θ₁/n)^(−θ₂)` | ✓ | ✓ |
| Muon g-2 Anomaly (Schwinger) | `θ₀ · (α/π)²` | ✓ | ✓ |
| Blackbody Planck Correction | structural match | — | ✓ |

## Project Structure

```
PhysicsPaper/
├── src/adcd/                       # Installable package
│   ├── __init__.py                 # Public API (adcd.fit, adcd.discover_correction)
│   ├── anomaly_scenarios.py        # 9 standard + 3 blind benchmark scenarios
│   ├── arc_scorer.py               # Asymptotic consistency gate (ARC)
│   ├── coarse_evaluator.py         # Coarse numerical pre-filter
│   ├── correction_orchestrator.py  # Main multi-iteration discovery loop
│   ├── dimensional_checker.py      # Dimensional homogeneity gate
│   ├── jax_optimizer.py            # JAX L-BFGS-B optimizer (parameter-scaled)
│   ├── llm_proposer.py             # Mock + Gemini + OpenAI-compatible proposers
│   ├── metrics.py                  # NMSE, BIC, structural classification
│   ├── pipeline.py                 # Stage 1 filter cascade
│   ├── real_data_loader.py         # Real-world data loading utilities
│   ├── real_scenarios.py           # Real-world validation scenarios
│   ├── residual_analyzer.py        # Statistical residual feature extraction
│   └── result.py                   # CorrectionResult: summary, LaTeX, plot
├── tests/                          # 58 unit + integration tests
├── paper/                          # LaTeX source (main.tex) + figures
├── run_correction_discovery.py     # Standard 9-scenario benchmark runner
├── run_real_data_benchmark.py      # Real-world physical constants benchmark
├── run_reproducibility.py          # Multi-seed reproducibility study
├── run_ablation.py                 # Gate ablation study
├── run_pysr_baseline.py            # PySR comparison baseline
├── run_mlp_baseline.py             # MLP comparison baseline
├── generate_figures.py             # Paper figure generator
├── .github/workflows/              # CI (test + lint + LaTeX) and PyPI publish
├── pyproject.toml                  # PEP 517/518 build configuration
└── README.md                       # This file
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest --cov=adcd
```

All 58 tests pass on Python 3.10 and 3.11 (Ubuntu and Windows).

## Reproducing Paper Results

```bash
python run_correction_discovery.py --proposer mock   # Main 9-scenario benchmark
python run_real_data_benchmark.py                    # Real-world physical constants
python run_ablation.py                               # Gate ablation study
python run_pysr_baseline.py                          # PySR comparison
python run_mlp_baseline.py                           # MLP comparison
python run_reproducibility.py                        # Multi-seed reproducibility
python generate_figures.py                           # Generate all paper figures
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

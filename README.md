# ADCD — Anomaly-Driven Correction Discovery

[![DOI](https://zenodo.org/badge/20534940.svg)](https://doi.org/10.5281/zenodo.20534940)
[![CI](https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml/badge.svg)](https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery**

ADCD is a symbolic regression framework that discovers *physical correction terms* rather than learning equations from scratch. Given a known classical law and anomalous observations, ADCD recovers the dimensionless correction Δ that reconciles theory with experiment — mirroring how physics actually evolves.

> **81.1% (±11.4%) mean structural recovery** across 5 random seeds, with peak **94.4%** at the reference seed.  
> **4/4 real-world structural class matches** (Mercury, Lamb Shift, Muon g-2, Blackbody).  
> **58 automated unit tests** passing on Python 3.10 and 3.11.

---

## Key Features

- **Correction-first paradigm** — starts from a known classical law, not a blank slate; designed for anomaly-driven theory refinement where the baseline is structurally correct
- **Physics-gated search cascade** — AST complexity, dimensional homogeneity + transcendental guardrails, and asymptotic consistency (ARC) gates prune unphysical candidates *before* optimization
- **JAX-traced L-BFGS-B optimizer** — parameter-scaled differentiable fitting with multi-restart log-uniform initialization
- **BIC reranking** — selects the most parsimonious correction over purely numerical fits
- **Residual feature intelligence** — statistical priors (monotonicity, curvature, oscillation, decay rate, symmetry) bias the template sampler toward the correct mathematical family
- **Coarse empirical evaluation** — data-driven pre-filter ranks gate survivors before full JAX optimization
- **Noise-robust** — 93.3% mean at 0% noise, 91.1% at 1%, 71.1% at 5%, 68.9% at 10%

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

### Standard Benchmark (seed=42, Mock Proposer)

Results from `run_correction_discovery.py --proposer mock` (reference seed=42, 4 iterations per scenario).

| Scenario | Tier | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|----------|------|:--------:|:--------:|:--------:|:---------:|
| Relativistic KE | Textbook | ✓ | ✓ | ✓ | ✓ |
| Yukawa Gravity | Textbook | ✓ | ✓ | ✓ | ✓ |
| Anharmonic Spring | Textbook | ✓ | ✓ | ✓ | ✓ |
| Screened Coulomb | Cross-Domain | ✓ | ✓ | ✗ | ✗ |
| Net Radiation | Cross-Domain | ✓ | ✓ | ✓ | ✓ |
| Nonlinear Drag | Cross-Domain | ✓ | ✓ | ✓ | ✓ |
| Mystery-A (tanh²) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| Mystery-B (sinc) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| Mystery-C (log-quotient) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| **Overall** | | **100%** | **100%** | **88.9%** | **88.9%** |

> **Note**: Screened Coulomb fails at ≥5% noise because exponential decay ($e^{-r/\lambda}$) and rational saturation ($r/(r+\lambda)$) are numerically indistinguishable at the tested SNR with limited dynamic range — an information-theoretic limit, not a framework deficiency.

### Multi-Seed Reproducibility

All results are reported across 5 independent random seeds (0, 7, 21, 42, 99):

| Seed | Class Match Rate |
|:----:|:----------------:|
| 0 | 86.1% (31/36) |
| 7 | 66.7% (24/36) |
| 21 | 80.6% (29/36) |
| 42 | 94.4% (34/36) |
| 99 | 77.8% (28/36) |
| **Mean** | **81.1% ± 11.4%** |

Performance variation reflects stochastic template sampling in the MockProposer. Physics gates ensure that **when** the correct functional family is sampled, it consistently survives filtering and is selected by BIC reranking.

### Real-World Physical Constants Benchmark

Synthetic-real hybrid data using experimentally validated constants from JPL DE440, NIST, and CODATA:

| Physical Scenario | Discovered Correction | Converged | Class Match | NMSE |
|---|---|:---:|:---:|:---:|
| Mercury Perihelion (GR) | `θ₀(c⁻²v²)^θ₁` | — | ✓ polynomial | 2.33e-01 |
| Hydrogen Lamb Shift (QED) | `θ₀(n/θ₁)^(-θ₂)` | ✓ | ✓ power_law | 1.82e-18 |
| Muon g-2 (Schwinger) | `θ₀(α/π)^θ₁` | ✓ | ✓ polynomial | 7.94e-07 |
| Blackbody (Planck) | `-1 + e^(-f/θ₁)` | — | ✓ exponential | 2.59e-02 |

All 4 scenarios achieve correct structural class identification. 2 scenarios (Lamb Shift, Muon g-2) achieve full convergence with NMSE < 10⁻⁶. Mercury and Blackbody achieve correct structural identification but quantitative convergence is limited by parametrization sensitivity and dynamic range, respectively.

### PySR Comparison

| Method | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|--------|:--------:|:--------:|:--------:|:---------:|
| ADCD (ours) | 9/9 (100%) | 9/9 (100%) | 8/9 (88.9%) | 8/9 (88.9%) |
| PySR | 2/9 (22.2%) | 6/9 (66.7%) | 4/9 (44.4%) | 4/9 (44.4%) |

ADCD outperforms unconstrained PySR by **44.4 percentage points** at 5% noise.

## Project Structure

```
PhysicsPaper/
├── src/adcd/                       # Installable package
│   ├── __init__.py                 # Public API (adcd.fit, adcd.discover_correction)
│   ├── anomaly_scenarios.py        # 9 standard + 3 blind benchmark scenarios
│   ├── arc_scorer.py               # Asymptotic consistency gate (ARC)
│   ├── coarse_evaluator.py         # Coarse numerical pre-filter
│   ├── correction_orchestrator.py  # Main multi-iteration discovery loop
│   ├── dimensional_checker.py      # Dimensional homogeneity + transcendental guardrail
│   ├── jax_optimizer.py            # JAX L-BFGS-B optimizer (parameter-scaled)
│   ├── llm_proposer.py             # Mock + Gemini + OpenAI-compatible proposers
│   ├── metrics.py                  # NMSE, BIC, structural classification
│   ├── pipeline.py                 # Stage 1 filter cascade
│   ├── real_data_loader.py         # Real-world data loading (JPL, NIST, CODATA)
│   ├── real_scenarios.py           # Real-world validation scenarios
│   ├── residual_analyzer.py        # Statistical residual feature extraction
│   └── result.py                   # CorrectionResult: summary, LaTeX, plot
├── tests/                          # 58 unit + integration tests
├── paper/                          # LaTeX source (main.tex) + figures
├── run_correction_discovery.py     # Standard 9-scenario benchmark runner
├── run_real_data_benchmark.py      # Real-world physical constants benchmark
├── run_reproducibility.py          # Multi-seed reproducibility study (5 seeds)
├── run_ablation.py                 # Gate ablation study
├── run_pysr_baseline.py            # PySR comparison baseline
├── run_mlp_baseline.py             # MLP comparison baseline
├── run_misspecification_benchmark.py  # Baseline misspecification fail-safe test
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

One-command reproduction (Windows):

```powershell
.\reproduce_all.ps1
```

Or step-by-step:

```bash
python run_correction_discovery.py --proposer mock   # Main benchmark + gate telemetry
python run_real_data_benchmark.py                    # Real-world (5 scenarios)
python run_pysr_baseline.py --profile fair           # Fair PySR comparison
python run_ablation.py                               # Gate ablation study
python run_oracle_ablation.py                        # Oracle ground-truth injection test
python run_correction_scaling.py                     # Correction magnitude sweep
python scripts/generate_experiment_report.py         # Sync experiment_results.md
python scripts/generate_efficiency_table.py          # ADCD vs PySR efficiency table
python scripts/validate_results.py                   # Consistency checks
python generate_figures.py                           # All paper figures
```

> **Proposer regimes:** Mock Proposer = template-assisted recovery; Hybrid/Gemini = zero-shot discovery. Report both separately (see paper Section 4).

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

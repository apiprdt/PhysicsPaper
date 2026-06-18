# ADCD — Anomaly-Driven Correction Discovery

<p align="center">
  <img src="docs/assets/adcd_logo.svg" alt="ADCD Logo" width="160">
</p>

<p align="center">
  <strong>Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery</strong>
</p>

<p align="center">
  <a href="https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml"><img src="https://github.com/apiprdt/PhysicsPaper/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://pypi.org/project/adcd/"><img src="https://img.shields.io/pypi/v/adcd?color=teal" alt="PyPI Version"></a>
  <a href="https://doi.org/10.5281/zenodo.20534940"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20534940-blue" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/apiprdt/PhysicsPaper/actions"><img src="https://img.shields.io/badge/tests-95%20passing-brightgreen" alt="Tests Status"></a>
  <a href="https://pypi.org/project/adcd/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Support"></a>
</p>

<p align="center">
  <a href="https://apiprdt.github.io/PhysicsPaper/"><strong>Explore the Documentation »</strong></a>
  <br>
  <br>
  <a href="#quick-start">Quick Start</a>
  ·
  <a href="#key-features">Key Features</a>
  ·
  <a href="#benchmark-results">Benchmarks</a>
  ·
  <a href="https://colab.research.google.com/github/apiprdt/PhysicsPaper/blob/main/notebooks/adcd_demo.ipynb">Run in Colab</a>
</p>

> **Science rarely discovers from a blank slate — it corrects.**  
> ADCD automates the step between *anomaly* and *theory correction*, the same step that led from Newtonian gravity to General Relativity, from Dirac to QED, and from Rayleigh-Jeans to Planck.

<p align="center">
  <img src="docs/adcd_discovery.gif" alt="ADCD discovery pipeline animation" width="85%">
</p>

---

## ⚡ Key Features

- **Correction-First Paradigm** — Starts from a known classical law, not a blank slate. Focuses the search space on the discrepancy $\Delta$ between theory and experiment.
- **Cascaded Physics Gates** — AST complexity, dimensional homogeneity, transcendental guardrails, and asymptotic consistency (ARC) gates screen out unphysical candidates *before* running parameter-fitting.
- **JAX-Traced L-BFGS-B Optimizer** — Highly optimized parameter-scaled differentiable fitting with multi-restart log-uniform initialization.
- **BIC Model Selection** — Employs the Bayesian Information Criterion (BIC) to rank models, favoring simpler physical theories over overly complex numerical fits.
- **Residual Feature Intelligence** — Extracts mathematical features (monotonicity, curvature, oscillation, decay) from residuals to bias proposal templates.
- **Phase 2: Multivariable Discovery** — Buckingham Π group decomposition + per-variable Sequential ARC + variance-factorization separability detection for multi-input physical laws.
- **Real-World Validated** — Successfully identifies correct structural classes on Mercury's perihelion (GR), Lamb Shift (QED), Muon g-2 (Schwinger), and Blackbody (Planck).

---

## 📦 Installation

Install the stable package from PyPI:

```bash
pip install adcd
```

Or install from source:

```bash
git clone https://github.com/apiprdt/PhysicsPaper.git
cd PhysicsPaper
pip install -e ".[dev]"
```

Verify your installation:
```bash
pytest tests/
```

---

## 💻 Quick Start

### 1. High-Level Scientific API

Running ADCD on predefined physics benchmarks is extremely simple:

```python
import adcd

# 1. Load a pre-defined benchmark scenario (e.g. Relativistic Kinetic Energy)
scenarios = adcd.get_all_scenarios()
scenario = scenarios[0] 

# 2. Run discovery in a single line!
result = adcd.discover_correction(scenario, max_iterations=5, proposer="mock")

# 3. View the best fit
print(f"Discovered correction: {result.best_expr}")       # θ₀ * (v/c)**2
print(f"LaTeX representation:  {result.export_latex()}")   # \theta_0 \left(\frac{v}{c}\right)^2
print(f"Parameters:            {result.best_theta}")
print(f"BIC Score:             {result.best_bic:.2f}")

# 4. Plot residuals
result.plot_residuals()
```

### 2. Custom Experimental Datasets

For custom datasets, use the `adcd.fit` function:

```python
import numpy as np
import adcd

# Your custom data
x = np.linspace(1.0, 5.0, 100)
X = {"x": x}
y_classical = 2.0 * x
y_observed  = 2.0 * x + 0.5 * x**2   # True correction is 0.5 * x^2

# Run ADCD
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

---

## 📊 Benchmark Results

### 1. Standard Benchmark (seed=42, Mock Proposer)

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

### 2. PySR Comparison (fair profile: 100 iterations, 60s timeout)

| Method | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|--------|:--------:|:--------:|:--------:|:---------:|
| **ADCD** (ours, seed=42) | **9/9 (100%)** | **9/9 (100%)** | **8/9 (88.9%)** | **8/9 (88.9%)** |
| PySR fair | 4/9 (44.4%) | 5/9 (55.6%) | 1/9 (11.1%) | 5/9 (55.6%) |

ADCD outperforms PySR by **+77.8 percentage points** at 5% noise.

### 3. Phase 2: Multivariable Benchmark (v2.2.0)

| Scenario | Variables | ADCD Solved | Notes |
|----------|-----------|:-----------:|-------|
| Yukawa Mass-Ratio | m, M, r, r₀ | ✓ | Π groups: m/M, r/r₀ |
| Turbulent Drag | v, ρ, A, C_D | ✓ | Separable multiplicative |
| Coupled Oscillator | k, m, Ω, ω₀ | ✗ | Mixed functional form |
| Van der Waals MV | a, b, P, V, T | ✗ | Requires 3rd Π group |
| **Overall** | | **2/4 (50%)** | Baseline: 0/4 |

### 4. Real-World Physical Constants

Validation on historical anomalies using physical constants from JPL DE440, NIST, and CODATA:

| Physical Scenario | Discovered Correction | Converged | Class Match | NMSE |
|---|---|:---:|:---:|:---:|
| Mercury Perihelion (GR) | `θ₀·vc²` | — | ✓ polynomial | 1.11e-05 |
| Hydrogen Lamb Shift (QED) | `θ₀(n/θ₁)^(-θ₂)` | ✓ | ✓ power_law | 1.82e-18 |
| Muon g-2 (Schwinger) | `θ₀(α/π)^θ₁` | ✓ | ✓ polynomial | 7.94e-07 |
| Blackbody (Planck) | `-1 + e^(-f/θ₁)` | — | ✓ exponential | 2.59e-02 |

---

## 📁 Project Structure

```
PhysicsPaper/
├── src/adcd/                       # Installable package
│   ├── __init__.py                 # Public API (fit, discover_correction)
│   ├── anomaly_scenarios.py        # 9 standard + 3 blind + 4 multivariable scenarios
│   ├── arc_scorer.py               # Asymptotic consistency gate (ARC)
│   ├── buckingham_pi.py            # [Phase 2] Buckingham Π group engine
│   ├── coarse_evaluator.py         # Coarse numerical pre-filter
│   ├── correction_orchestrator.py  # Main multi-iteration discovery loop
│   ├── dimensional_checker.py      # Dimensional homogeneity + transcendental gate
│   ├── jax_optimizer.py            # JAX L-BFGS-B optimizer
│   ├── llm_proposer.py             # Mock + Gemini + OpenAI proposers
│   ├── metrics.py                  # NMSE, BIC, structural classification
│   ├── multivar_orchestrator.py    # [Phase 2] Multivariable correction pipeline
│   ├── pipeline.py                 # Stage 1 filter cascade
│   ├── real_data_loader.py         # Real-world data loading (JPL, NIST, CODATA)
│   ├── residual_factorizer_v2.py   # [Phase 2] Variance-decomposition separability
│   ├── result.py                   # CorrectionResult object
│   └── sequential_arc.py           # [Phase 2] Per-variable Sequential ARC checker
├── tests/                          # 95 unit + integration tests
├── paper/                          # LaTeX source (main.tex) + figures
├── run_correction_discovery.py     # Benchmark runner
└── README.md                       # This file
```

---

## 📖 Citing This Work

If you use ADCD in your research, please cite:

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained
                Symbolic Regression for Evolutionary Scientific Discovery}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.1.3},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```

---

## 👥 AI Disclosure

This project was developed with assistance from Google DeepMind's Antigravity AI assistant. AI was used as a pair-programming and writing tool. All scientific content, experimental design decisions, and intellectual contributions are the author's own.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

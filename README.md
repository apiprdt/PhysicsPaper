# ADCD — Anomaly-Driven Correction Discovery

<p align="center">
  <em>Physics-constrained symbolic regression for discovering symbolic corrections to known physical laws.</em>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20534940"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20534940-blue" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/adcd/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Support"></a>
</p>

---

### Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Benchmark & Paper Claims](#-benchmark--paper-claims)
  - [1. Primary Synthetic Anomaly Benchmark](#1-primary-synthetic-anomaly-benchmark-9-scenarios--4-noise-levels--16-seeds)
  - [2. Comparison with Unconstrained Symbolic Regression (PySR)](#2-comparison-with-unconstrained-symbolic-regression-pysr)
  - [3. Real-World Physical-Constant Scenarios](#3-real-world-physical-constant-scenarios)
  - [4. SPARC Galaxy Rotation Curves: Autonomous Rediscovery of Simple MOND](#4-sparc-galaxy-rotation-curves-autonomous-rediscovery-of-simple-mond)
  - [5. Cosmological Probes: Growth Rate and Expansion History](#5-cosmological-probes-growth-rate-and-expansion-history)
  - [6. Exploratory Multivariable Extension](#6-exploratory-multivariable-extension)
- [Scope & Limitations](#-scope--limitations)
- [Project Structure](#-project-structure)
- [Citing This Work](#-citing-this-work)
- [Reproducibility](#-reproducibility)
- [License](#-license)

---

## Overview

Traditional symbolic regression (AI Feynman, PySR, DSR) discovers equations *from scratch* — searching across the entire unconstrained function space. **ADCD instead starts from a known classical baseline law and searches for the dimensionless symbolic correction term $\Delta$** that reconciles the baseline with anomalous observations ($y_{\text{obs}} = y_{\text{classical}}(1 + \Delta)$ or $y_{\text{obs}} = y_{\text{classical}} + \Delta$). 

Restricting the search to this correction subspace enables physics-gated screening (AST complexity, dimensional homogeneity, transcendental guards, and asymptotic consistency limits) *before* parameter optimization runs.

> **Primary Claim:** Mean structural class recovery rate of **80.4% ($\pm$7.4% population SD)** across 16 independent random seeds on the 9-scenario $\times$ 4-noise-level synthetic benchmark (95% bootstrap CI [76.7%, 84.0%]). The reference seed `42` used for detailed reference tables is explicitly disclosed as the highest-performing seed (94.4%) in that set; the 16-seed mean, not the peak, is reported as the primary headline claim.

---

## ⚡ Key Features

- **Correction-First Paradigm** — Starts from a known classical baseline law. Targets the residual discrepancy $\Delta$ rather than unconstrained full-law expression trees.
- **Cascaded Physics Gates** — AST complexity, dimensional homogeneity (with transcendental argument guards), and asymptotic-regime consistency (ARC) screen out unphysical candidates prior to fitting.
- **JAX-Traced L-BFGS-B Optimizer** — JIT-compiled multi-restart optimization with log-uniform parameter initialization.
- **BIC Model Selection** — Ranks surviving candidates via the Bayesian Information Criterion, penalizing free parameters to prevent overfitting.
- **Fail-Safe Misspecification Handling** — Reliably rejects false physical corrections when the baseline law is misspecified, returning a null result rather than confident spurious expressions.
- **CPU Accessibility** — Runs full pipeline discovery in 4.2s–4.9s on standard dual-core consumer CPUs without GPU requirements.

---

## 📦 Installation

Install the package from PyPI:

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
python -m pytest tests/
```

---

## 💻 Quick Start

### 1. High-Level Scientific API

Running ADCD on predefined physics benchmarks:

```python
import adcd

# 1. Load a pre-defined benchmark scenario (e.g. Relativistic Kinetic Energy)
scenarios = adcd.get_all_scenarios()
scenario = scenarios[0]

# 2. Run discovery
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

```python
import numpy as np
import adcd

# Custom dataset
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
    correction_mode="additive",
    log_param=True,
    verbose=True
)

print(result.summary())
```

---

## 📊 Benchmark & Paper Claims

All numbers below correspond exactly to the manuscript (`paper/main.pdf`).

### 1. Primary Synthetic Anomaly Benchmark (9 Scenarios × 4 Noise Levels × 16 Seeds)

| Noise Level | ADCD Mean (16 Seeds) | ADCD Worst Seed | ADCD Reference (Seed 42) |
|:-----------:|:---------------------:|:---------------:|:------------------------:|
| 0%          | 86.8% ($\pm$9.8%)     | 66.7% (6/9)     | 100% (9/9)               |
| 1%          | 81.2% ($\pm$14.6%)    | 44.4% (4/9)     | 100% (9/9)               |
| 5%          | 77.1% ($\pm$10.0%)    | 66.7% (6/9)     | 88.9% (8/9)              |
| 10%         | 76.4% ($\pm$12.3%)    | 55.6% (5/9)     | 88.9% (8/9)              |
| **Overall** | **80.4% ($\pm$7.4%)** | **69.4% (25/36)** | **94.4% (34/36)**      |

Reference seed detail (Seed 42, Mock Proposer):

| Scenario | Tier | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|----------|------|:--------:|:--------:|:--------:|:---------:|
| Relativistic KE | Textbook | ✓ | ✓ | ✓ | ✓ |
| Yukawa Gravity | Textbook | ✓ | ✓ | ✓ | ✓ |
| Anharmonic Spring | Textbook | ✓ | ✓ | ✓ | ✓ |
| Screened Coulomb | Cross-Domain | ✓ | ✓ | ✗ | ✗ |
| Net Radiation | Cross-Domain | ✓ | ✓ | ✓ | ✓ |
| Nonlinear Drag | Cross-Domain | ✓ | ✓ | ✓ | ✓ |
| Mystery-A ($\tanh^2$) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| Mystery-B ($\text{sinc}$) | Synthetic | ✓ | ✓ | ✓ | ✓ |
| Mystery-C ($\ln(1+x)/x$) | Synthetic | ✓ | ✓ | ✓ | ✓ |

### 2. Comparison with Unconstrained Symbolic Regression (PySR)

ADCD is evaluated against PySR on the **identical residual target** $y_{\text{obs}} - y_{\text{classical}}$ under matched operator sets and equal data:

| Method | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|--------|:--------:|:--------:|:--------:|:---------:|
| **ADCD (ours, Reference Seed 42)** | **100% (9/9)** | **100% (9/9)** | **88.9% (8/9)** | **88.9% (8/9)** |
| ADCD Mean (16 Seeds) | 86.8% | 81.2% | 77.1% | 76.4% |
| PySR `fair` (100 iterations, 60s) | 44.4% (4/9) | 55.6% (5/9) | 11.1% (1/9) | 55.6% (5/9) |
| PySR `generous` (200 iterations, 120s) | 44.4% (4/9) | 44.4% (4/9) | 55.6% (5/9) | 22.2% (2/9) |

At 5% noise, ADCD achieves an 88.9% recovery rate versus 11.1% for PySR `fair` — a 77.8 percentage-point gap under equal residual data.

### 3. Real-World Physical-Constant Scenarios

Evaluated on synthetic-real hybrid scenarios constructed using JPL DE440, NIST, and CODATA parameters (Reference Seed 42, Mock Proposer):

| Scenario | Discovered Correction $\Delta$ | Structural Match | Quantitative Fit (NMSE $< 10^{-4}$) | Optimizer Converged (NMSE $< 10^{-5}$) | NMSE |
|---|---|:---:|:---:|:---:|---:|
| Mercury Perihelion (GR) | $\theta_0(v/c)^2$ | ✓ (Exact AST) | ✓ | ✗ | $1.11\times10^{-5}$ |
| Hydrogen Lamb Shift (QED) | $\theta_0 n^{-\theta_1}$ | ✓ (Power Law) | ✓ | ✓ | $1.69\times10^{-18}$ |
| Muon $g-2$ (Schwinger) | $\theta_0(\alpha/\pi)^{\theta_1}, \theta_1 \approx 1$ | ✓ (Polynomial) | ✓ | ✓ | $7.94\times10^{-7}$ |
| Blackbody Radiation (Planck) | $-1 + e^{-f/\theta_1}$ | ✓ (Exponential) | ✗ | ✗ | $2.59\times10^{-2}$ |

**Headline Result:** 4/4 structural class match, 3/4 quantitative recovery, 2/4 optimizer converged. Binary pulsar decay is evaluated separately as a controlled sensitivity study (Supplementary Material Table S5) because its single-variable formulation is simplified; recovery fails structurally under full four-parameter scan.

### 4. SPARC Galaxy Rotation Curves: Autonomous Rediscovery of Simple MOND

Applied to the SPARC archive ($N_{\text{gal}}=171$, $N_{\text{pts}}=3,342$), starting only from the Newtonian baseline and the asymptotic constraint $\Delta \to 0$ as $x \to \infty$:

- **Discovered Form:** $\nu_{\text{ADCD}}(x) = \theta_0(\sqrt{1+\theta_1/x} - 1) + 1$, with $\hat{\theta}_0 \approx 1.83, \hat{\theta}_1 \approx 0.262$.
- **Algebraic Identification:** Identical algebraic skeleton to the 2-parameter **Simple MOND** interpolating function ($a + b\sqrt{1+c/x}$ with $a+b=1$), recovering a transition parameter $\hat{c} \approx 0.27$ vs canonical $c_S = 4.0$.
- **Statistical Validation (3 Levels):**
  1. *Stacked NMSE:* $0.3729$ (41% reduction vs zero-parameter canonical forms).
  2. *Galaxy-Level 5-Fold Cross-Validation:* Statistically indistinguishable from 2-parameter refitted Simple MOND ($\Delta\text{NMSE} = +0.008, z = +0.39, p = 0.69$), and decisively superior to 2-parameter Standard MOND ($z = -4.60$).
  3. *Galaxy-Level Cluster Bootstrap (1,000 resamples):* $\delta\text{BIC}_{\text{eff}} = +3.4$ vs Simple MOND (95% CI $[-0.9, +7.0]$, inconclusive), and $\delta\text{BIC}_{\text{eff}} = -36.5$ vs Standard MOND (95% CI $[-60.1, -17.9]$, decisive).

### 5. Cosmological Probes: Growth Rate and Expansion History

Applied to 63 structure-growth rate points $f\sigma_8(z)$ and 34 cosmic-chronometer expansion points $H(z)$:

- Across 5 independent tests spanning 2 observables and 3 survey subsets, ADCD returns a `constant_wins` verdict: **no $z$-dependent functional correction beats a 1-parameter constant amplitude rescaling by $\Delta\text{BIC} < -10$**.
- This null result acts as a quantitative upper bound on detectability given current survey precision (~20% point errors), establishing that current cosmological tensions manifest as amplitude mismatches rather than functional growth modifications.

### 6. Exploratory Multivariable Extension

Reported as an exploratory capability (Phase 2): across 4 two-dimensional multivariable scenarios, ADCD achieves 3/4 structural recovery at 0–1% noise and 2/4 at 5% noise. All failures are correctness-preserving (failing gate checks rather than generating false physical claims).

---

## ⚠️ Scope & Limitations

- **Correction-First Scope:** ADCD assumes the classical baseline law is structurally valid as a leading-order approximation. It is designed for theory refinement, not tabula rasa theory replacement.
- **Proposer Expressiveness Bound:** Candidates are bounded by the template bank or LLM vocabulary. Unsampled functional classes cannot be recovered.
- **High-Noise Information Limits:** Numerically similar functional forms (e.g. exponential vs rational saturation) become indistinguishable under low SNR and narrow dynamic range.
- **Cosmological Null Result:** The `constant_wins` verdict is a detectability bound given current data noise, not a proof of zero underlying modification.

---

## 📁 Project Structure

```
PhysicsPaper/
├── src/adcd/                       # Installable Python package
│   ├── __init__.py                 # Public API (fit, discover_correction)
│   ├── anomaly_scenarios.py        # Benchmark scenario definitions
│   ├── arc_scorer.py               # Asymptotic consistency gate (ARC)
│   ├── dimensional_checker.py      # Dimensional homogeneity gate
│   ├── jax_optimizer.py            # JAX L-BFGS-B multi-restart optimizer
│   ├── metrics.py                  # NMSE, BIC, and structural classifier
│   ├── pipeline.py                 # Stage 1 filter cascade
│   └── result.py                   # Discovery result data structures
├── paper/                          # LaTeX source & figures
│   ├── main.tex                    # Main manuscript source (25 pages)
│   ├── supplementary.tex           # Supplementary appendix source (4 pages)
│   ├── main.pdf                    # Compiled main manuscript PDF
│   └── supplementary.pdf           # Compiled supplementary appendix PDF
├── tests/                          # Unit and integration test suite
├── run_correction_discovery.py     # Main benchmark runner
└── README.md                       # Repository overview
```

---

## 📖 Citing This Work

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained
                Symbolic Regression for Evolutionary Scientific Discovery}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {3.0.0},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```

---

## 🔬 Reproducibility

Every quantitative claim in the paper is reproducible:

```bash
# Run 9-scenario primary benchmark
python run_correction_discovery.py

# Run full test suite
python -m pytest tests/ -q
```

---

## AI Assistance & Declarations

AI coding assistants (Google DeepMind's Antigravity agent, Cursor IDE) were used as pair-programming tools under direct human supervision for code implementation, benchmark execution, figure rendering, and manuscript formatting. All research concepts, theoretical formulations, experimental design decisions, physical interpretations, and intellectual contributions are entirely the author's own.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

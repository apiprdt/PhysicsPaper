# ADCD — Anomaly-Driven Correction Discovery

<p align="center">
  <em>Physics-constrained symbolic regression for discovering symbolic corrections to known physical laws.</em>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20534940"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20534940-blue" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/adcd/"><img src="https://img.shields.io/pypi/v/adcd" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/JAX-accelerated-orange" alt="JAX">
</p>

---

## What is ADCD?

Most symbolic regression tools start from scratch. ADCD takes a different approach: **you already know the classical law** (`y_classical`), and ADCD finds the symbolic correction term `Δ` that explains the residual anomaly.

```
y_observed = y_classical × (1 + Δ)    # multiplicative correction
y_observed = y_classical + Δ           # additive correction
```

ADCD enforces physical constraints at every step — dimensional homogeneity, classical limit recovery, and structural parsimony — so the discovered correction is physically meaningful, not just a curve fit.

---

## Installation

```bash
pip install adcd
```

Or from source (recommended for development):

```bash
git clone https://github.com/apiprdt/PhysicsPaper.git
cd PhysicsPaper
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, SymPy, JAX, Matplotlib, scikit-learn, pandas.

---

## Quick Start

### 1. Use a Built-in Benchmark Scenario

```python
import adcd

# Load a pre-defined scenario (22 scenarios available)
scenarios = adcd.get_all_scenarios()
scenario = scenarios[0]  # Relativistic KE

# Run discovery (mock proposer — no API key needed)
result = adcd.discover_correction(
    scenario,
    max_iterations=5,
    proposer="mock",   # or "gemini" / "hybrid" (requires GEMINI_API_KEY)
    seed=42,
    verbose=True,
)

# Physicist-friendly summary
print(result.summary(brief=True))
```

**Terminal output:**
```
[ADCD Auto-Mode] Detected multiplicative correction with confidence 0.97
  [####################] Iter 1/5  |  25 proposed -> 9 passed (36%)  |  NMSE: 5.44e-31  BIC: -2757.8  |  best: theta_0 * v**2

  [CONVERGED] Iteration 1  |  NMSE = 5.44e-31  |  R^2 = 100.0%  |  Quality: Excellent

+==========================================================================+
| ADCD  ·  Correction Discovery Results                                    |
|==========================================================================|
| Scenario   : Relativistic KE                                             |
| Domain     : relativistic mechanics                                      |
| Type       : multiplicative  │  Limit: v → 0                             |
| Status     : ✓ CONVERGED  (iteration 1 of 1)                             |
|==========================================================================|
| DISCOVERED CORRECTION                                                    |
|--------------------------------------------------------------------------|
|   Δ = theta_0 * v**2                                                     |
|   LaTeX: \Delta = \theta_{0} v^{2}                                       |
|   Substituted: 8.333e-18 * v**2                                          |
|--------------------------------------------------------------------------|
| FITTED PARAMETERS                                                        |
|   theta_0 = 8.3333333e-18                                                |
|==========================================================================|
| FIT QUALITY                                                              |
|--------------------------------------------------------------------------|
|   Variance explained  : [####################] 100.0%                    |
|   Quality             : Excellent  ✓✓                                    |
|   Success criterion   : NMSE_res < 0.20  ->  PASS v                      |
+==========================================================================+
```

### 2. Fit Your Own Data

```python
import numpy as np
import adcd

# Your data: classical prediction and observations
x = np.linspace(1.0, 5.0, 100)
X = {"x": x}
y_classical = 2.0 * x
y_observed  = 2.0 * x + 0.5 * x**2   # anomaly: + 0.5 * x^2

result = adcd.fit(
    X=X,
    y_obs=y_observed,
    y_classical=y_classical,
    limit_variable="x",
    limit_direction="0",      # correction vanishes as x → 0
    correction_mode="additive",
    proposer="mock",
    verbose=True,
)

print(result.summary())        # full dual-layer output
print(result.export_latex())   # LaTeX string for the paper
result.show_candidates(top_k=5)  # ranked candidate table
result.plot()                  # residual + reconstruction plots
```

### 3. Inspect Results Programmatically

```python
result.best_expr          # "theta_0 * x**2"
result.best_theta         # {"theta_0": 0.5}
result.best_nmse_residual # e.g. 1.2e-28
result.r_squared          # 0.9999...
result.fit_quality_label  # "Excellent"
result.converged          # True
result.latex              # LaTeX string from SymPy
result.export_latex()     # full \Delta = ... string
repr(result)              # ADCDResult(expr='theta_0 * x**2', R²=1.0000, ...)
```

---

## Available Scenarios (22 total)

```python
scenarios = adcd.get_all_scenarios()
for s in scenarios:
    print(f"  {s.name:35s} [{s.domain}]")
```

| # | Scenario | Domain |
|---|---|---|
| 1 | Relativistic KE | relativistic mechanics |
| 2 | Yukawa Gravity | gravitation |
| 3 | Anharmonic Spring | mechanics |
| 4 | Screened Coulomb | electrostatics |
| 5 | Net Radiation | thermodynamics |
| 6 | Nonlinear Drag | fluid dynamics |
| 7–9 | Mystery-A / B / C | gravitation / mechanics |
| 10–18 | Blind-1 … Blind-9 | van der Waals, Stokes-Einstein, Wien, relativistic pendulum, Casimir, … |
| 19–22 | MV-1 … MV-4 | multivariable (Yukawa mass-ratio, plasma, turbulence, van der Waals 2D) |

---

## Proposer Modes

| Proposer | Description | Requires |
|---|---|---|
| `"mock"` | Rule-based template bank, deterministic, no internet | Nothing |
| `"gemini"` | LLM-guided proposal via Gemini API | `GEMINI_API_KEY` |
| `"hybrid"` | Mock bank + LLM refinement (recommended) | `GEMINI_API_KEY` |

```python
import os
os.environ["GEMINI_API_KEY"] = "your-key-here"

result = adcd.discover_correction(scenario, proposer="hybrid")
```

---

## How It Works

ADCD applies a 3-stage cascaded gate pipeline before optimization:

```
Candidate expressions
        │
  ┌─────▼──────┐
  │  Stage 1   │  AST Validator  — syntax & safety check
  │  (Filter)  │  Dimensional Checker — physical units consistent?
  │            │  ARC Scorer — correct classical limit (v→0, x→0, …)?
  └─────┬──────┘
        │ survivors (~30–40% of candidates)
  ┌─────▼──────┐
  │  Stage 2   │  JAX L-BFGS-B optimizer (15 restarts, multi-seed)
  │  (Optimize)│  BIC ranking — fewer parameters, same fit = better
  └─────┬──────┘
        │ best candidate
  ┌─────▼──────┐
  │  Stage 3   │  Bayesian reranker — posterior weight over candidates
  │  (Rank)    │  Identifiability analysis — parameter uniqueness check
  └─────┬──────┘
        │
   ADCDResult
```

**Success criterion:** `NMSE_res < 0.20` (≥ 80% of residual variance explained by the discovered correction).

---

## Benchmark Performance

Performance measured at **seed=42, 0% noise**, on the 9 standard synthetic scenarios:

| Scenario | Domain | Result | NMSE_res |
|---|---|---|---|
| Relativistic KE | rel. mechanics | ✅ polynomial | 5.4e-31 |
| Yukawa Gravity | gravitation | ✅ exponential | 3.3e-13 |
| Screened Coulomb | electrostatics | ✅ exponential | 2.0e-17 |
| Net Radiation | thermodynamics | ✅ power_law | 1.9e-12 |
| Nonlinear Drag | fluid dynamics | ✅ polynomial | 8.6e-32 |
| Mystery-A / B | gravitation / mechanics | ✅ trigonometric | ~6e-15 |
| Mystery-C | mechanics | ✅ logarithmic | 2.8e-12 |
| Anharmonic Spring | mechanics | ❌ (mock proposer gap) | 1.0 |

**Mean structural recovery: 8/9 = 88.9%** (seed=42, mock proposer, 0% noise).

Across 16 independent seeds and 4 noise levels (0%, 1%, 3%, 5%), the mean structural recovery rate is **80.4% ± 7.4%**, outperforming unconstrained PySR by **77.8 pp** at 5% noise.

> **Note on success criterion**: `class_match = True` requires **both** (a) correct structural class discovered AND (b) `NMSE_res < 0.20`. Both conditions must hold simultaneously.

---

## Real-World Application: SPARC Galaxy Rotation Curves

ADCD was applied to 171 SPARC galaxy rotation curves (N = 3,342 data points). Without any domain guidance, it autonomously rediscovered a **2-parameter member of the Simple MOND algebraic family**:

```
Δ_MOND = (a_0 / g_classical)^(1/2)
```

- Stacked NMSE = 0.3729 (moderate; galaxy scatter dominates)
- Statistically consistent with domain-expert MOND forms under galaxy-level cross-validation
- Cosmological probes (fσ₈/H(z)): `constant_wins` null result — no significant correction found (honest negative result)

---

## Repository Structure

```
PhysicsPaper/
├── src/adcd/                         # Installable Python package
│   ├── __init__.py                   # Public API
│   ├── api.py                        # fit() and discover_correction()
│   ├── result.py                     # ADCDResult with dual-layer output
│   ├── display.py                    # Terminal box drawing & HTML helpers
│   ├── anomaly_scenarios.py          # 22 benchmark scenario definitions
│   ├── correction_orchestrator.py    # Main search loop
│   ├── arc_scorer.py                 # Asymptotic Regime Checker (ARC)
│   ├── dimensional_checker.py        # Dimensional homogeneity gate
│   ├── jax_optimizer.py              # JAX L-BFGS-B multi-restart optimizer
│   ├── metrics.py                    # NMSE, BIC, structural classifier
│   ├── pipeline.py                   # Stage 1 filter cascade
│   ├── bayesian_ranker.py            # Posterior candidate reranking
│   ├── identifiability.py            # Parameter identifiability analysis
│   └── llm_proposer.py              # Mock / Gemini / Hybrid proposers
├── paper/                            # LaTeX manuscript
│   ├── main.tex / main.pdf           # Main paper (25 pages)
│   ├── supplementary.tex/.pdf        # Supplementary appendix
│   └── figures/                      # All paper figures (PDF)
├── tests/                            # 234-test suite (pytest)
├── scripts/verify_paper_claims.py    # Reproduce paper benchmark numbers
├── run_real_data_benchmark.py        # SPARC real-data pipeline runner
└── README.md
```

---

## Reproducibility

```bash
# Run the full test suite (234 tests, ~5 min)
python -m pytest tests/

# Verify primary benchmark numbers from the paper
python scripts/verify_paper_claims.py

# Run the real-data SPARC benchmark (requires SPARC dataset)
python run_real_data_benchmark.py
```

**SPARC data:** Download `Rotmod_LTG.zip` from [sparc.astro.cwru.edu](http://sparc.astro.cwru.edu) and set:
```bash
export SPARC_DATA_DIR="/path/to/Rotmod_LTG"
```

---

## Limitations (Honest)

- **Mock proposer**: Template-based; may miss novel structures not in the bank. Use `"hybrid"` for best results on unknown data.
- **Anharmonic Spring gap**: The mock proposer's template bank currently does not cover the specific polynomial correction for this scenario. The `"hybrid"` proposer resolves this.
- **Cosmological data**: ADCD returns a `constant_wins` null on fσ₈/H(z) — suggesting the anomaly structure is not expressible as a simple symbolic correction to ΛCDM with the current proposal grammar.
- **Execution time**: 4–5 seconds per run (mock, single core). Hybrid proposer adds LLM latency (~10–30s).
- **SPARC fit quality**: Stacked NMSE = 0.37 reflects real galaxy scatter, not algorithm failure. Individual galaxies vary significantly.

---

## Citation

If you use ADCD in your research, please cite the Zenodo software release:

```bibtex
@software{adcd2026,
  title     = {{ADCD}: Anomaly-Driven Correction Discovery —
               Physics-Constrained Symbolic Regression for
               Evolutionary Scientific Discovery},
  year      = {2026},
  publisher = {Zenodo},
  version   = {3.0.0},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```

---

## AI Assistance Declaration

AI coding assistants (Google DeepMind Antigravity, Cursor IDE) were used as pair-programming tools for code implementation, benchmark execution, figure rendering, and manuscript formatting under direct human direction. All research concepts, theoretical formulations, experimental design decisions, physical interpretations, and intellectual contributions are entirely the author's own.

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

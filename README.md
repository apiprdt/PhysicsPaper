<div align="center">

# Asymptotic Dictionary Correction Discovery (ADCD)

*A deterministic, identifiability-aware framework for recovering algebraic corrections to known physical laws from noisy observational data.*

<p>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/google/jax"><img alt="JAX Float64" src="https://img.shields.io/badge/JAX-Float64-red.svg"></a>
  <a href="#reproducibility"><img alt="Byte-exact" src="https://img.shields.io/badge/results-byte--exact-brightgreen.svg"></a>
</p>

</div>

---

## ⚡ Quick Start (5 minutes)

> Reproduce all three validated scenarios from the paper with one command.

**Step 1 — Clone & install**

```bash
git clone <repo-url>
cd PhysicsPaper

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

pip install -r requirements.txt
```

**Step 2 — Run the full validation**

```bash
# Linux / macOS
export PYTHONPATH=src
python src/adcd/run_adcd_v3_validation_blind.py --top-k 5

# Windows PowerShell
$env:PYTHONPATH = "src"
python src\adcd\run_adcd_v3_validation_blind.py --top-k 5
```

**Step 3 — Read the output**

Results print to stdout and are also saved automatically to `run_outputs/adcd_v3_taxonomy_validation_report.json`.

Expected summary:

```
═══════════════════════════════════════════════
 ADCD v3  —  Validation Summary
═══════════════════════════════════════════════
 Scenario          NMSE       ΔBIC   Verdict
───────────────────────────────────────────────
 Screened Coulomb  2.77e-04   30.65  IDENTIFIABLE
 Entropy Expansion 1.59e-02   14.70  IDENTIFIABLE
 Time Dilation     4.12e-01   13.79  WITHHELD
═══════════════════════════════════════════════
```

> **Reproducibility:** Results are byte-exact across runs (deterministic grammar enumeration + fixed seed). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## 📋 Table of Contents

1. [How ADCD Works](#how-adcd-works)
2. [Installation (detailed)](#installation-detailed)
3. [Python API](#python-api)
   - [Run a built-in scenario](#run-a-built-in-scenario)
   - [Define your own scenario](#define-your-own-scenario)
   - [Call the pipeline directly](#call-the-pipeline-directly)
   - [Inspect results](#inspect-results)
   - [Use individual components](#use-individual-components)
4. [Validated Results](#validated-results)
5. [Repository Structure](#repository-structure)
6. [Reproducibility](#reproducibility)
7. [Limitations](#limitations)
8. [Citation & License](#citation--license)

---

## How ADCD Works

**Core idea:** Given a known classical law $y_\text{cl}$ and noisy observations $y_\text{obs}$, discover the algebraic correction $\Delta$ such that:

$$y_\text{obs} = y_\text{cl} \cdot (1 + \Delta), \quad \lim_{u \to 0} \Delta = 0$$

The constraint $\lim_{u \to 0}\Delta = 0$ (the correction vanishes in the classical regime) is enforced **by construction** — no post-hoc filtering.

```
Classical law y_cl  ─────┐
                          ├─► Residual ──► Grammar ──► Gates ──► Optimizer ──► Pareto ──► Verdict
Observations y_obs  ─────┘   (Δ = y_obs/y_cl - 1)
```

### The 5 Physical Primitives

All corrections are expressed over five basis functions, each vanishing at $u = 0$ by construction:

| Name | $D(u)$ | Physical archetype |
|:---|:---|:---|
| `D_lor` | $u / [\sqrt{1-u}\,(\sqrt{1-u}+1)]$ | Lorentz / special relativity |
| `D_rat` | $u / (1-u)$ | Rational poles |
| `D_exp` | $e^{-u} - 1$ | Exponential screening (Debye, Yukawa) |
| `D_log` | $\ln(1+u)$ | Entropy, Van der Waals |
| `D_sqrt_inv` | $1/\sqrt{1+u} - 1$ | Inverse-root corrections |

### The 4-Step Validation Protocol

| Step | Name | Criterion |
|:---|:---|:---|
| **1** | Budget Disclosure | Search space size logged *before* results |
| **2** | Positive Control | Optimizer recovers structure within correct primitive family (NMSE < 0.05) |
| **3** | Ablation Control | $\Delta\text{BIC} > 10$ vs. best model without the correct primitive |
| **4** | Determinism Check | 3 independent runs produce byte-exact identical JSON output |

A scenario is **IDENTIFIABLE** only if all four steps pass. Otherwise it is **WITHHELD** — meaning the pipeline honestly reports insufficient evidence.

---

## Installation (detailed)

**Requirements:** Python 3.10+, CPU only (no GPU needed).

```bash
# Clone the repository
git clone <repo-url>
cd PhysicsPaper

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

# Install dependencies (pinned to exact paper versions)
pip install -r requirements.txt

# Set the source path (required before any import or command)
export PYTHONPATH=src              # Linux / macOS
# $env:PYTHONPATH = "src"         # Windows PowerShell
```

To verify installation:

```bash
python -c "from adcd.pipeline import Stage1Pipeline; print('ADCD installed OK')"
```

---

## Python API

### Run a built-in scenario

```python
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import run_scenario_protocol

# Get all three paper scenarios
scenarios = get_all_scenarios()
sc = next(s for s in scenarios if s.name == "Screened Coulomb")

# Run the full 4-step blind protocol
results = run_scenario_protocol(sc, top_k=5)
print(results["verdict"])   # "IDENTIFIABLE"
```

### Define your own scenario

For real observational data (CSV), ADCD provides a strictly deterministic ingestion layer (`auto_scenario.py`) backed by `pint` for robust physical unit parsing:

```python
from adcd.auto_scenario import build_scenario_from_csv

# Loads CSV, parses physical units automatically (e.g., from column 'velocity [km/s]'),
# checks dimensions, and calculates residuals safely.
scenario = build_scenario_from_csv(
    csv_path="data/my_observational_data.csv",
    scenario_name="Custom Orbit Anomaly",
    target_col="acceleration [m/s**2]",
    classical_expr="velocity**2 / radius",
    domain="gravity_orbital",  # strictly validated against DOMAIN_TAXONOMY
    classical_limit_variable="velocity",
)

# Extract ready-to-use numpy arrays
X, y_obs, y_classical, residual = scenario.generate_data()
```

For purely synthetic testing, you can construct an `AnomalyScenario` directly:

<details>
<summary><strong>Show synthetic scenario example</strong></summary>

```python
from adcd.anomaly_scenarios import AnomalyScenario

scenario = AnomalyScenario(
    name="My Scenario",
    tier="synthetic",
    domain="mechanics",
    classical_expr="0.5 * m * v**2",
    classical_variables=["m", "v"],
    classical_constants={"c": 2.99792458e8},
    correction_type="multiplicative",
    correction_expr="theta_0 * (v/c)**2",
    correction_constants={"theta_0": 0.5},
    anomaly_regime="high speeds v approaching c",
    variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
    classical_limit_variable="v",
    classical_limit_direction="0",
    correction_class="rational",
)

X, y_obs, y_classical, residual = scenario.generate_data(n_points=200, noise_level=0.01, seed=42)
```
</details>

### Call the pipeline directly

```python
from adcd.pipeline import Stage1Pipeline
from adcd.grammar_proposer_v3 import GrammarProposerV3
from adcd.dimensional_checker import DimensionalChecker
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.context import ProposalContext

# Build pipeline components
checker  = DimensionalChecker()
arc      = ARCScorer(regimes=build_arc_regimes())
context  = ProposalContext.from_scenario(scenario)

# Enumerate candidates (fully blind — no ratio hints)
proposer   = GrammarProposerV3()
candidates = proposer.propose(context)
print(f"Search space: {len(candidates)} candidates")

# Run all 5 physical gates
pipeline  = Stage1Pipeline(checker, scenario)
survivors = pipeline.run(candidates, X, y_obs, y_classical)
print(f"Survived all gates: {len(survivors)} candidates")
```

### Inspect results

```python
results = run_scenario_protocol(scenario, top_k=5)

print(f"Verdict:         {results['verdict']}")
print(f"Top candidate:   {results['blind_search']['top_candidate']}")
print(f"NMSE:            {results['blind_search']['nmse']:.4e}")
print(f"BIC:             {results['blind_search']['bic']:.2f}")
print(f"ΔBIC (ablation): {results['ablation_control']['bic_diff']:.2f}")

# Full Pareto front
for rank, c in enumerate(results['pareto_front'], 1):
    print(f"  Rank {rank} | BIC {c['bic']:>10.2f} | NMSE {c['nmse']:.3e} | {c['expr']}")
```

**Example output (Screened Coulomb):**

```
Verdict:         IDENTIFIABLE
Top candidate:   theta_2 * (exp(-r/theta_0) - 1.0)
NMSE:            2.7722e-04
BIC:             -1617.66
ΔBIC (ablation): 30.65

  Rank 1 | BIC   -1617.66 | NMSE 2.77e-04 | theta_2 * (exp(-r/theta_0) - 1.0)
  Rank 2 | BIC   -1612.93 | NMSE 2.76e-04 | theta_47 * (exp(-r/theta_0) - 1.0) * (...)
```

### Use individual components

<details>
<summary><strong>Dimensional Checker</strong></summary>

```python
from adcd.dimensional_checker import DimensionalChecker
import sympy as sp

checker = DimensionalChecker()
checker.registry["v"] = [1, 0, -1]   # m/s → [L, M, T]
checker.registry["c"] = [1, 0, -1]

v, c = sp.symbols("v c")
is_ok, dims = checker.check_dimensionless(v**2 / c**2)
print(is_ok, dims)   # True  [0, 0, 0]
```
</details>

<details>
<summary><strong>ARC Scorer (Asymptotic Regime Check)</strong></summary>

```python
from adcd.arc_scorer import ARCScorer, build_arc_regimes
import sympy as sp

arc = ARCScorer(regimes=build_arc_regimes())
u   = sp.Symbol("u")

print(arc.passes(u / (1 - u), limit_var=u, limit_val=0))           # True  ✓ vanishes
print(arc.passes(sp.Rational(1, 2) + u, limit_var=u, limit_val=0)) # False ✗ non-zero
```
</details>

<details>
<summary><strong>JAX Optimizer (standalone fit)</strong></summary>

```python
from adcd.jax_optimizer import JaxOptimizer
import sympy as sp, numpy as np

optimizer = JaxOptimizer(n_restarts=15)
expr      = sp.sympify("theta_0 * (exp(-r / theta_1) - 1)")

params, nmse = optimizer.fit(expr, X={"r": r_data}, y=residual_data)
print(f"Fitted: {params},  NMSE: {nmse:.4e}")
```
</details>

<details>
<summary><strong>BIC & Identifiability Verdict</strong></summary>

```python
from adcd.identifiability import compute_bic, identifiability_verdict

bic     = compute_bic(nmse=2.77e-4, n_params=2, n_points=200)
verdict = identifiability_verdict(delta_bic=30.65)  # "IDENTIFIABLE" | "WITHHELD"
print(verdict)
```
</details>

---

## Validated Results

Three scenarios, $N = 200$ points, 1% Gaussian noise, `seed = 42`.  
All numbers from a live JAX run on CPU — see `run_outputs/`.

| Scenario | Observation window | Rank-1 structure | ΔBIC | NMSE | Verdict |
|:---|:---|:---|---:|---:|:---|
| Screened Coulomb | $r \le 4.0\,\text{m}$ | $\theta_0(e^{-r/\theta_1} - 1)$ | 30.65 | 2.77 × 10⁻⁴ | **IDENTIFIABLE** |
| Entropy Expansion | $dV/V_i \le 1.0$ | $\theta_0\ln(1 + dV/V_i)$ | 14.70 | 1.59 × 10⁻² | **IDENTIFIABLE** |
| Time Dilation | $v \le 0.3c$ | $\theta_0 \cdot D_\text{lor}(v^2/c^2)$ | 13.79 | 4.12 × 10⁻¹ | **WITHHELD** |

> Time Dilation: Lorentz structure is found at Rank 1 and ΔBIC > 10, but the **positive control fails** (NMSE = 0.41 > 0.05) within the historical $v \le 0.3c$ window — meaning there is insufficient signal-to-noise for confident identification. The WITHHELD verdict is by design: the pipeline does not claim identifiability it cannot demonstrate.

---

## Repository Structure

```
PhysicsPaper/
├── src/adcd/                              ← Core library
│   ├── run_adcd_v3_validation_blind.py    ← Main entry point (4-step protocol)
│   ├── anomaly_scenarios.py               ← AnomalyScenario dataclass + data generation
│   ├── pipeline.py                        ← 5-gate physical filter pipeline
│   ├── grammar_proposer_v3.py             ← Deterministic candidate enumeration
│   ├── asymptotic_dictionary_proposer_v3.py ← Buckingham-Pi ratio derivation
│   ├── arc_scorer.py                      ← Gate 4: asymptotic limit check (ARC)
│   ├── dimensional_checker.py             ← Gates 1–3: AST / units / transcendental
│   ├── coarse_evaluator.py                ← Gate 5: NaN / inf pre-filter
│   ├── jax_optimizer.py                   ← Float64 JAX L-BFGS-B optimizer
│   ├── jax_precision_config.py            ← JAX float64 precision enforcement
│   ├── bayesian_ranker.py                 ← BIC ranking and Pareto front builder
│   ├── identifiability.py                 ← ΔBIC computation and verdict logic
│   ├── metrics.py                         ← NMSE, symbolic + class match scoring
│   ├── constants.py                       ← CODATA 2018 physical constants
│   ├── context.py                         ← ProposalContext: scenario → grammar input
│   ├── auto_scenario.py                   ← Deterministic CSV ingestion with pint units
│   ├── quickfit.py                        ← Convenience single-fit wrapper
│   ├── budget_sweep.py                    ← Search-space budget analysis
│   ├── mode_detection.py                  ← Residual mode detection utilities
│   ├── residual_features.py               ← Residual feature extraction
│   ├── real_data_loader.py                ← Loader for external observational data
│   └── real_scenarios.py                  ← Non-synthetic scenario definitions
│
├── eval/                                  ← Evaluation & ablation scripts
│   ├── benchmark_runner.py
│   ├── ablation_extended.py
│   ├── independent_evaluator.py
│   └── compute_metrics.py
│
├── paper/                                 ← Paper source (LaTeX + figures)
│   ├── neurips_paper.tex                  ← Main manuscript
│   ├── neurips_2026.sty                   ← NeurIPS 2026 style file
│   ├── references.bib                     ← BibTeX bibliography
│   ├── fig1_recovery.pdf                  ← Figure 1: Correction recovery plots
│   ├── fig2_bic.pdf                       ← Figure 2: BIC identifiability
│   ├── fig3_parity.pdf                    ← Figure 3: Parity plots
│   └── generate_final_3_figures.py        ← Figure generation script
│
├── run_outputs/                           ← Validation reports (JSON)
├── data/                                  ← Reference datasets
├── tests/                                 ← Unit tests
│
├── reproduce_all.sh                       ← One-command full reproduction (Linux/macOS)
├── reproduce_all.ps1                      ← One-command full reproduction (Windows)
├── requirements.txt                       ← Pinned dependencies
├── pyproject.toml                         ← Package metadata
└── REPRODUCIBILITY.md                     ← Full provenance chain
```

---

## Reproducibility

```bash
# Full reproduction — results saved to run_outputs/
export PYTHONPATH=src
python src/adcd/run_adcd_v3_validation_blind.py --top-k 5

# Or use the convenience script
bash reproduce_all.sh        # Linux / macOS
# .\reproduce_all.ps1        # Windows PowerShell
```

Results are **byte-exact** across runs because:
- Grammar enumeration is fully deterministic (no stochastic mutations)
- All random seeds are fixed (`seed=42` throughout)
- JAX computation graph is `jit`-compiled on CPU with `float64` precision

For the full provenance chain — including a-priori AST budget justification and cross-run determinism verification — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## Limitations

| Issue | Detail |
|:---|:---|
| **Multiplicative corrections only** | Grammar targets $y = y_\text{cl}(1 + \Delta)$. Additive corrections require different residual normalization. |
| **Synthetic benchmarking** | The core validations use synthetic noise for strict ground-truth comparison. Real data is supported via `auto_scenario.py`, but extensive real-world benchmarking is ongoing. |
| **Window-dependent verdicts** | Identifiability depends on the observation regime. Results are specific to the stated domain bounds. |
| **Charge / temperature dimensionless** | The dimensional checker treats these as `[0,0,0]` by convention; ratios involving them need manual registry extension. |
| **Positive-control threshold** | Step 2 uses NMSE < 0.05. Scenarios with high inherent noise or very flat residuals may require threshold tuning. |

---

## Citation & License

```bibtex
@inproceedings{adcd2026,
  title     = {Asymptotic Dictionary Correction Discovery: A Deterministic,
               Identifiability-Aware Framework for Recovering Physical Corrections
               from Observational Data},
  author    = {Anonymous Author(s)},
  booktitle = {New in ML Workshop @ NeurIPS 2026},
  year      = {2026},
}
```

Released under the **MIT License** — see [LICENSE](LICENSE).

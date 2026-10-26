<div align="center">

# ADCD: Asymptotic Dictionary Correction Discovery for Identifiable Symbolic Regression

*A deterministic, identifiability-aware framework for recovering algebraic corrections to known physical laws from noisy observational data.*

**[NeurIPS 2026 Presentation](https://neurips.cc/virtual/2026/loc/paris/180051)** · **[Paper (PDF)](paper/main.pdf)** · **[Quickstart Notebook](notebooks/01_quickstart.ipynb)**

<p>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/google/jax"><img alt="JAX Float64" src="https://img.shields.io/badge/JAX-float64-red.svg"></a>
  <a href="#reproducibility"><img alt="Byte-exact" src="https://img.shields.io/badge/results-byte--exact-brightgreen.svg"></a>
</p>

</div>

---

## What is ADCD?

Most physics equations are *corrections to a known baseline*: Einstein did not throw away Newton — he found the extra term that matters at high speeds. ADCD automates that same reasoning.

Given:
- A **classical baseline** (e.g. Coulomb's law)
- **Noisy observations** that deviate from it
- A **taxonomy key** naming the physics domain

ADCD searches for the symbolic correction term `Δ` that:
1. Vanishes exactly at the classical limit (algebraic guarantee — no floating-point collapse)
2. Is dimensionally consistent (5D MLTΘQ Buckingham-π engine)
3. Issues a formal **WITHHELD** verdict instead of a hallucinated answer when data cannot support a structural claim

**Three mechanisms, one pipeline:**

```
Noisy data → [Mode detection] → [Gate cascade] → [JAX optimizer] → IDENTIFIABLE / WITHHELD
                                      ↑
              Algebraically regularized primitives + Buckingham-π + Taxonomy prior
```

---

## Results at a Glance

| Scenario | Window | ΔBIC | NMSE | Verdict |
|---|---|---|---|---|
| Screened Coulomb | r ≤ 4.0 m | 30.65 | 2.77 × 10⁻⁴ | **IDENTIFIABLE** (exact match) |
| Entropy Expansion | dV/Vᵢ ≤ 1.0 | 14.70 | 1.59 × 10⁻² | **IDENTIFIABLE** (class match) |
| Time Dilation | v ≤ 0.3c | 13.79 | 0.41 | **WITHHELD** (signal too weak) |

All numbers are byte-exact reproducible. Time Dilation being WITHHELD is *intended behavior*, not a failure — it means the 4.8% correction is indistinguishable from 1% noise at that observation window.

---

## Quick Start (5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/apiprdt/PhysicsPaper.git
cd PhysicsPaper
```

Create a virtual environment:

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package and dependencies:

```bash
pip install -e ".[dev]"
# Or strictly pinned lockfile:
pip install -r requirements.txt
```

### 2. Run via the `adcd` CLI

Once installed with `pip install -e .`, the `adcd` CLI is available everywhere:

```bash
# Run all three canonical scenarios
adcd validate --top-k 5

# Or run a single scenario
adcd validate --scenario "Screened Coulomb"

# Inspect built-in scenarios, domain taxonomy priors, and regularized primitives
adcd list-scenarios
adcd list-taxonomy
adcd list-primitives
```

*(Alternatively, without installing the package entrypoints, run directly via Python:)*

```bash
# Linux / macOS
export PYTHONPATH=src
python src/adcd/run_adcd_v3_validation_blind.py --top-k 5

# Windows PowerShell
$env:PYTHONPATH = "src"
python src\adcd\run_adcd_v3_validation_blind.py --top-k 5
```

Expected output summary:

```
══════════════════════════════════════════════════════
 ADCD v3  —  Validation Summary
══════════════════════════════════════════════════════
 Scenario            NMSE       ΔBIC    Verdict
──────────────────────────────────────────────────────
 Screened Coulomb    2.77e-04   30.65   IDENTIFIABLE
 Entropy Expansion   1.59e-02   14.70   IDENTIFIABLE
 Time Dilation       4.12e-01   13.79   WITHHELD
══════════════════════════════════════════════════════
```

Full JSON results are saved to `run_outputs/adcd_v3_taxonomy_validation_report.json`.

### 3. Interactive notebook

For a step-by-step walkthrough of one experiment:

```bash
pip install jupyter
jupyter notebook notebooks/01_quickstart.ipynb
```

---

## Repository Structure

```
PhysicsPaper/
├── src/adcd/                  # Core library
│   ├── anomaly_scenarios.py   # Three canonical experiment definitions
│   ├── pipeline.py            # Five-gate pre-fitting cascade
│   ├── jax_optimizer.py       # L-BFGS-B optimizer (JAX float64)
│   ├── quickfit.py            # Domain taxonomy (DOMAIN_TAXONOMY dict)
│   ├── dimensional_checker.py # 5D MLTΘQ Buckingham-π engine
│   ├── arc_scorer.py          # Asymptotic Regularity Check (ARC)
│   ├── metrics.py             # Extended BIC, NMSE, structure classification
│   ├── auto_scenario.py       # CSV → AnomalyScenario ingestion layer
│   └── run_adcd_v3_validation_blind.py  # Main validation entrypoint
├── notebooks/
│   └── 01_quickstart.ipynb    # End-to-end demo (Screened Coulomb)
├── paper/
│   ├── main.tex               # LaTeX source (NeurIPS 2026 template)
│   ├── main.pdf               # Compiled paper (camera-ready)
│   └── references.bib         # Bibliography
├── rebuttal/                  # 3×3 misspecification audit tables
├── run_outputs/               # Saved JSON validation reports
├── tests/                     # pytest test suite
├── requirements.txt           # Pinned dependency lockfile
├── pyproject.toml             # Package metadata
├── CONTRIBUTING.md            # How to add scenarios and extend the taxonomy
└── REPRODUCIBILITY.md         # Number-by-number provenance log
```

---

## Running a Single Scenario

You can run ADCD on a single scenario programmatically:

```python
import adcd

# Load the Screened Coulomb scenario
scenarios = {s.name: s for s in adcd.get_all_scenarios()}
scenario = scenarios["Screened Coulomb"]

# Generate synthetic data (ground truth hidden from ADCD)
X, y_classical, y_obs, delta_true = scenario.generate_data(
    n_points=200,
    noise_level=0.01,  # 1% Gaussian noise
    seed=42,
    domain_max=4.0,    # historical window: r ≤ 4.0 m
)

print(f"ADCD version: {adcd.__version__}")
print(f"Scenario:     {scenario.name}")
print(f"Domain:       {scenario.domain}")
print(f"N points:     {len(X['r'])}")
```

See `notebooks/01_quickstart.ipynb` for the full pipeline including optimizer and verdict.

---

## Adding a Custom Scenario

**Option A — Python API** (recommended for reproducibility):

1. Add a new entry to `DOMAIN_TAXONOMY` in `src/adcd/quickfit.py` with peer-reviewed primitive justification.
2. Create an `AnomalyScenario` instance in `src/adcd/anomaly_scenarios.py`.
3. Run `src/adcd/run_adcd_v3_validation_blind.py`.

**Option B — CSV ingestion** (fastest for exploratory use):

```python
from adcd.auto_scenario import build_scenario_from_csv

scenario = build_scenario_from_csv(
    csv_path="my_data.csv",          # columns: variable names with [unit] headers
    scenario_name="My Experiment",
    target_col="y_obs [V]",          # column name for the noisy observation
    classical_expr="k_e * q1 * q2 / r",
    domain="yukawa_debye_screening", # must match a key in DOMAIN_TAXONOMY
    classical_constants={"k_e": 8.98755e9},
    classical_limit_variable="r",
    classical_limit_direction="0",
)
```

CSV header format: `variable_name [unit]`, e.g. `radius [m]`, `charge_1 [C]`.

See `CONTRIBUTING.md` for the full taxonomy extension protocol.

---

## Reproducibility

All results in the paper are byte-exact deterministic. To verify:

```bash
# Linux / macOS
bash reproduce_all.sh

# Windows PowerShell
.\reproduce_all.ps1
```

The scripts run all three scenarios three times and assert that all runs are byte-identical.
Number-by-number provenance is in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Citing

If you use ADCD in your research, please cite:

```bibtex
@inproceedings{erdita2026adcd,
  title     = {Asymptotic Dictionary Correction Discovery: A Deterministic,
               Identifiability-Aware Framework for Recovering Physical
               Corrections from Observational Data},
  author    = {Erdita, Muhammad Afif},
  booktitle = {NewInML Workshop at the 40th Conference on Neural Information
               Processing Systems (NeurIPS 2026)},
  year      = {2026},
  address   = {Paris, France},
  url       = {https://neurips.cc/virtual/2026/loc/paris/180051},
  note      = {NeurIPS 2026 NewInML Workshop, Paper \#180051},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built independently by Muhammad Afif Erdita · Indonesia · 2026</sub>
</div>

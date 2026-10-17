# Contributing to ADCD

Thank you for your interest in ADCD. This guide explains how to add a new physics scenario,
extend the taxonomy, and run the test suite.

---

## Development Setup

```bash
git clone https://github.com/apiprdt/PhysicsPaper.git
cd PhysicsPaper

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -e ".[dev]"
```

## Running Tests

```bash
export PYTHONPATH=src   # Windows: $env:PYTHONPATH = "src"
pytest tests/ -q
```

---

## Adding a New Scenario

### Step 1 — Justify the taxonomy entry

Open `src/adcd/quickfit.py` and locate the `DOMAIN_TAXONOMY` dictionary. Add a new key:

```python
"my_phenomenon": ["D_exp", "D_log"],
# Justification: Boltzmann partition function → e^{-βε} (exponential);
# entropy of mixing → ln(1+x) (logarithmic).
# Citation: Callen (1985), Thermodynamics, Chapter 3.
#
# Rule: only add a primitive if a peer-reviewed paper on THIS phenomenon
# uses that functional form in a first-principles derivation.
```

> **Do not** add a primitive because a fit looks better — that is overfitting the taxonomy.
> A broad prior will shrink the BIC gap and make everything IDENTIFIABLE regardless of
> data quality, defeating the purpose of the WITHHELD verdict.

### Step 2 — Define the scenario

In `src/adcd/anomaly_scenarios.py`, add an `AnomalyScenario` instance:

```python
AnomalyScenario(
    name="My Scenario",
    tier="textbook",             # "textbook" | "synthetic"
    domain="my_phenomenon",      # must exactly match a DOMAIN_TAXONOMY key
    classical_expr="k * x",      # known baseline (SymPy-parseable string)
    classical_variables=["x"],
    classical_constants={"k": 1.0},
    correction_type="multiplicative",   # or "additive"
    correction_expr="theta_0 * x**2",  # ground truth — hidden from the pipeline
    correction_constants={"theta_0": 0.01},
    anomaly_regime="large x",
    variables_with_units={"x": "[m]"},
    classical_limit_variable="x",
    classical_limit_direction="0",      # correction → 0 as x → 0
    correction_class="power_law",       # for evaluation labelling only
)
```

### Step 3 — Implement the data-generation branch

In the `generate_data` method of `AnomalyScenario`, add an `elif` branch for your scenario:

```python
elif self.name == "My Scenario":
    x_max = domain_max if domain_max is not None else 5.0
    X["x"] = rng.uniform(0.1, x_max, size=n_points)
```

### Step 4 — Run the four-step protocol

```bash
export PYTHONPATH=src
python src/adcd/run_adcd_v3_validation_blind.py --top-k 5
```

Record the output in `REPRODUCIBILITY.md` with:
- The exact command used
- The random seed
- All raw headline numbers (ΔBIC, NMSE, verdict)

### Step 5 — Write a test

Add a test in `tests/` following the pattern in `tests/test_auto_scenario.py`.

---

## Code Style

| Tool | Config |
|---|---|
| Formatter | `black` (line length 120) |
| Linter | `flake8` |

```bash
black src/ && flake8 src/
```

---

## Taxonomy Philosophy

The Phenomenon-Specific Taxonomy is the core epistemic claim of ADCD. Its correctness
depends entirely on primitive sets being derived from first principles, never from fit quality.

> **Rule:** A new primitive may be added to a domain **only if** a peer-reviewed paper on
> that specific phenomenon uses that functional form in a first-principles derivation.
> The citation must appear as a comment next to the taxonomy entry.

Violating this rule makes WITHHELD verdicts meaningless. If the prior is too broad, a noisy
candidate can always fit, the BIC gap collapses, and the system declares everything
IDENTIFIABLE regardless of whether the data actually support a structural claim.

---

## Four-Step Validation Protocol (summary)

When you add a new scenario, ADCD's validation entrypoint checks four things in order:

| Step | Name | Pass condition |
|---|---|---|
| 1 | Budget disclosure | |C| announced before any fitting |
| 2 | Positive control | NMSE < 0.05 when restricted to the true family |
| 3 | Ablation control | ΔBIC > 10 over the ablated (truth-excluded) search |
| 4 | Determinism | Three runs produce byte-identical output |

Failure of any step yields **WITHHELD**, not an error. That is the correct behavior.

---

## Questions

Open an issue on [GitHub](https://github.com/apiprdt/PhysicsPaper/issues).

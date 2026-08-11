<div align="center">

# Asymptotic Dictionary Correction Discovery (ADCD)

*A deterministic, identifiability-aware framework for recovering algebraic corrections to known physical laws from noisy observational data.*

<p>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/google/jax"><img alt="JAX Float64" src="https://img.shields.io/badge/JAX-Float64-red.svg"></a>
</p>

</div>

---

## ⚡ Quick Start (Reproducing the Paper Results)

You can reproduce the core findings (Time Dilation, Screened Coulomb, Entropy Expansion) with a single command. 
This process performs a strictly **blind search** without target leakage, uses dynamic structure guessing, and enforces rigorous Asymptotic Regime Constraints (ARC) post-fitting.

### Step 1: Environment Setup

```bash
git clone <repo-url>
cd PhysicsPaper

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate          
# On Windows PowerShell:
# .\.venv\Scripts\Activate.ps1    

pip install -r requirements.txt
```

### Step 2: Run Full Validation Suite

```bash
# On Linux/macOS
export PYTHONPATH=src
python src/adcd/run_adcd_v3_validation_blind.py --taxonomy

# On Windows PowerShell
$env:PYTHONPATH = "src"
python src\adcd\run_adcd_v3_validation_blind.py --taxonomy
```
*Note: The `--taxonomy` flag instructs the search pipeline to leverage the Bayesian taxonomy prior for robust candidate generation.*

### Step 3: Generate Figures

After the validation script finishes, it produces a JSON report in `run_outputs/`. You can immediately generate the publication-ready PDF figures (Figures 1, 2, and 3):

```bash
python paper/generate_final_3_figures.py
```
The figures will be saved in the `paper/` directory as `fig1_recovery.pdf`, `fig2_bic.pdf`, and `fig3_ablation.pdf`.

---

## 🧠 What is ADCD?

ADCD (*Asymptotic Dictionary Correction Discovery*) is a symbolic regression framework specifically engineered for physics. Unlike black-box neural networks or unconstrained genetic algorithms, ADCD forces discovered mathematical expressions to obey known extreme physical limits (e.g., Lorentz factors collapsing to 1 at $v \to 0$).

### Core Features

1. **Dimensional Homogeneity Enforcement:** Rejects dimensionally invalid combinations (e.g., adding meters to seconds) before they are ever evaluated.
2. **Asymptotic Regime Constraints (ARC):** Automatically enforces that any discovered correction decays into the known classical limit in historical operational bounds. Any expression failing numerical ARC evaluation post-fitting is severely penalized ($BIC \to \infty$).
3. **Identifiability Guardrails:** Uses Kass-Raftery thresholds ($\Delta BIC \geq 10$) to explicitly differentiate between statistically justified discoveries and over-parameterized hallucinations. If the evidence is weak, ADCD outputs "WITHHELD" rather than guessing.
4. **Blind Search & Mode Detection:** The algorithm dynamically detects whether a perturbation is additive or multiplicative directly from the data distribution, eliminating target leakage.

---

## 📂 Repository Structure

- `src/adcd/` — Core source code of the ADCD framework.
- `src/adcd/run_adcd_v3_validation_blind.py` — The primary entry point for reproducing the paper's three validation scenarios.
- `paper/` — Scripts and assets for reproducing publication figures.
- `run_outputs/` — Auto-generated JSON reports containing Pareto fronts and full model evaluations.

## 📜 License

This project is released anonymously for double-blind peer review and workshops.

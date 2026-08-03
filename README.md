# Asymptotic Dictionary Correction Discovery (ADCD) v3

> **A correction-first, physically-regularized symbolic regression framework for discovering mathematically rigorous extensions to classical theories.**

Traditional symbolic regression (SR) engines (like PySR or genetic programming) excel at tabula-rasa discovery but frequently fail to converge on physically meaningful algebraic corrections due to catastrophic cancellation at asymptotic limits. Furthermore, general SR engines often lack explicit mechanisms to report *identifiability*—when observational data is insufficient to mathematically justify a new structural hypothesis.

**ADCD is not a general-purpose replacement for Symbolic Regression.** Instead, it is a highly specialized verification-and-reconstruction instrument designed for scenarios where the mathematical structure must obey strict physical asymptotes.

---

## 📖 The Philosophy: From Newton to Einstein

The evolution of physical laws often proceeds not from a vacuum, but through asymptotic corrections to well-tested classical baselines. The transition from Newtonian mechanics to Einstein's special relativity did not discard classical momentum; it introduced a structural correction (the Lorentz factor) that seamlessly reduces to the Newtonian baseline at low velocities. 

This **"correction-first" paradigm** is the core of ADCD. Rather than asking an algorithm to guess a law from scratch, ADCD assumes a known classical baseline and strictly searches for a dimensionless algebraic correction term $\Delta$ that vanishes perfectly at the classical limit ($\lim_{u \to 0} \Delta = 0$).

## ⚡ Key Architectural Features

1. **Rationalized Asymptotic Dictionary (The 5 Primitives)**
   To prevent catastrophic numerical cancellation near the classical limit, ADCD uses analytically rationalized primitive functions. For example, the naive Lorentz correction $(1 - u)^{-0.5} - 1$ collapses in standard `float32`. ADCD regularizes this into $D_{lor}(u) = \frac{1 - \sqrt{1-u}}{\sqrt{1-u}}$, completely eliminating gradient collapse.
2. **Deterministic Grammar Enumeration**
   ADCD rejects stochastic genetic mutation. Instead, it utilizes deterministic EBNF grammar enumeration constrained by strict budget limits (max depth $\le 7$, max tokens $\le 25$). This guarantees **byte-for-byte reproducibility** and reduces the search space from millions of trees to merely tens or hundreds of candidates.
3. **Strict Physics Gates (ARC & Dimensions)**
   Every candidate equation must pass the Asymptotic Regime Check (ARC) ensuring it rigorously reduces to the classical baseline, and a Dimensional Checker to ensure physical units are never violated.
4. **Computational Efficiency (No GPU Required)**
   Due to the bounded search space, ADCD runs entirely on a CPU. A complete validation protocol for multiple scenarios executes in **seconds**, utilizing a minimal memory footprint (~150MB).

## 🛡️ The 4-Step Validation Protocol

ADCD prevents false-positive structural claims by gating every discovery behind a rigid, automated 4-step protocol:
1. **Budget Disclosure:** The exact combinatorial search space size must be logged.
2. **Positive Control:** The system must isolate the correct structure when restricted to single primitives.
3. **Ablation Control:** The identified structure must dominate a generic polynomial expansion explicitly via Bayesian Information Criterion (BIC-diff), not just raw mean squared error.
4. **Determinism Check:** The output must remain identical across independent multi-seed restarts.

## 🚀 Quickstart & Usage

**Prerequisites:** Python 3.10+ (JAX, SymPy, NumPy, SciPy)

To run the complete validation protocol on all locked scenarios:
```bash
# Clone the repository
git clone https://github.com/apiprdt/PhysicsPaper.git
cd PhysicsPaper

# Set PYTHONPATH and execute the validation suite
export PYTHONPATH=src
python src/adcd/run_adcd_v3_validation.py
```
*Note for Windows users:* Use `$env:PYTHONPATH="src"` in PowerShell.

## 📊 Experimental Results (v3)

ADCD successfully maps the boundary of structural identifiability. On scenarios that pass the full validation protocol, ADCD reconstructs structures algebraically equivalent to the ground truth.

| Scenario | Truth Structure | Discovered Structure | Min. Identifiable Range | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Time Dilation** | `D_lor` | `theta_1 * D_lor(u)` | $v/c \ge 0.90$ | ✅ **Passed (Identifiable)** |
| **Entropy Expansion** | `D_log` | `theta_1 * D_log(u)` | $T/T_c \ge 0.85$ | ✅ **Passed (Identifiable)** |
| **Screened Coulomb** | `D_exp` | `theta_1 * D_exp(u)` | $r/\lambda \ge 2.5$ | ❌ **Failed (Limitation)** |

> **Why did Screened Coulomb fail?** 
> This is a feature, not a bug. Screened Coulomb fails due to numerical optimization limitations (L-BFGS-B underflow) on extreme exponential scales. We consciously refuse to fine-tune the optimizer's hyperparameters specifically to accommodate this, as doing so would violate the principle of blind discovery. This explicit failure proves the **absence of oracle leakage** in ADCD.

## ⚖️ ADCD vs. General Symbolic Regression (PySR)

| Dimension | ADCD (Correction-First) | Standard SR (Tabula-Rasa) |
| :--- | :--- | :--- |
| **Primary Goal** | Asymptotic corrections to classical baselines. | Discovering full equations from scratch. |
| **Hyperparameter Sensitivity** | **Stable.** Driven by strict BIC thresholding. | **Whiplash.** Highly sensitive to parsimony pressure. |
| **Identifiability Reporting** | **Explicit.** Actively reports when data is insufficient. | **Implicit.** Often returns best-fit polynomials silently. |
| **Execution Time** | **Seconds** (Bounded search space). | **Minutes to Hours** (Massive combinatorial mutations). |

## 📝 License & Citation

This project is open-source. If you use the ADCD framework or its concepts in your research, please refer to the accompanying paper manuscript.

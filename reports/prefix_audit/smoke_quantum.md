# Phase 0b Smoke Test Report: Quantum Known-Answer Cases Through ADCD v1

**Date:** 2026-11-08  
**Branch:** `ExploreSmth`  
**Execution:** Pinned CPU float64 environment (Python 3.11.9, JAX 0.10.1, NumPy 2.4.3)  
**Script:** `scripts/smoke_quantum.py`

---

## 1. Executive Summary

This report documents the execution of ADCD v1 on two canonical quantum-statistical known-answer transitions:
1. **Planck vs Rayleigh–Jeans Radiation:** $\Delta_{\text{true}}(x) = \frac{x}{e^x - 1} - 1$ across windows $x \in [0.01, 0.3]$, $[0.01, 1.0]$, and $[0.01, 3.0]$.
2. **Einstein Heat Capacity vs Dulong–Petit:** $\Delta_{\text{true}}(x) = \frac{x^2 e^x}{(e^x - 1)^2} - 1$ across windows $x \in [0.01, 0.3]$ and $[0.01, 1.0]$.

All runs used synthetic $N=200$ datasets with $1\%$ relative Gaussian noise ($\epsilon \sim \mathcal{N}(0, 0.01^2)$) evaluated under the full ADCD v1 blind search pipeline.

---

## 2. Predicted vs. Actual Metrics

| Scenario | Max Correction | $\Delta\chi^2$ vs Null | Noise-Floor NMSE | Best Primitives / Candidates Found | v1 Verdict | Discrepancy / Explanation |
|---|:---:|:---:|:---:|---|:---:|---|
| **Planck $x \le 0.3$** | 14.3% | 18,145 | 0.0539 | `log(1+x)` (NMSE 4.61e-2, BIC -599.65)<br>`tanh(x)` (NMSE 4.62e-2, BIC -599.25)<br>`exp(-x)-1` (NMSE 4.65e-2, BIC -598.13) | **WITHHELD** | **Matches Prediction:** Forms a tight equivalence class ($\Delta\text{BIC} < 1.5$). Ablation fails because rival primitives fit equally well. |
| **Planck $x \le 1.0$** | 41.8% | 291,320 | 0.0043 | `log(1+x) + (1-cos(x))` (NMSE 3.67e-3, BIC -1100.30)<br>`log(1+x) + (1/sqrt(1+x)-1)` (NMSE 3.69e-3, BIC -1099.66)<br>`exp(-x)-1 * (1 + |x|^θ)` (NMSE 3.66e-3, BIC -1095.64) | **WITHHELD** | **Matches Prediction:** Equivalence class persists ($\Delta\text{BIC} \approx 4.6$ between compositions). v1 ablation gate fails to isolate a single primitive. |
| **Planck $x \le 3.0$** | 84.3% | 1.07e7 | 0.0005 | **Rank 1:** `(exp(-x)-1) * (1 + θ_1 |x|^θ_2)` (NMSE 5.16e-4, BIC -1487.39)<br>**Rank 2:** `(exp(-x)-1) * (1 + θ_1 log(1+x))` (NMSE 6.11e-4, BIC -1459.13)<br>**Best non-exp:** `|x|^θ * (1 + 1/sqrt(1+x)-1)` (NMSE 8.67e-4, BIC -1383.74) | **WITHHELD** | **Exponential Family Wins Decisively:** Ablation $\Delta\text{BIC} = 103.11 \gg 10$. However, v1's positive control evaluates only the isolated single primitive (`exp(-x)-1` without secondary terms), which yields NMSE 0.0607 > 0.05, causing a false veto. |
| **Einstein $x \le 0.3$** | 0.75% | 23.5 | 20.04 | `rat` (NMSE 0.973, BIC +10.18)<br>`lor` (NMSE 0.973, BIC +10.26) | **WITHHELD** | **Matches Prediction:** Signal is completely swamped by noise (NMSE floor 20.0). Correctly withheld. |
| **Einstein $x \le 1.0$** | 7.9% | 2,966 | 0.1657 | `1 - cos(x)` (NMSE 0.131, BIC -390.29)<br>`1/sqrt(1+x^2)-1` (NMSE 0.130, BIC -386.95)<br>`exp(-x^2)-1` (NMSE 0.130, BIC -386.69) | **WITHHELD** | **Matches Prediction:** Signal is detectable ($\Delta\chi^2 \approx 2966$), but parity mismatch (intrinsically $x^2$) and noise floor prevent reaching NMSE < 0.05. |

---

## 3. Core Architectural Questions Answered

### Q1: Does the engine build the argument itself?
**YES.**
In `src/adcd/quickfit.py` (`_generate_ratio_symbols`), the engine derives dimensionless arguments via a two-pass heuristic:
1. **Pass 0 (Dimensionless inputs):** Any input marked `"dimensionless"` or `"1"` automatically generates linear and quadratic ratios: `[x, x**2]`.
2. **Pass 1 (Same-dimension pairs):** For pairs of variables sharing the exact same 5D dimension vector, it generates exact ratios ($a/b, b/a, a^2/b^2, b^2/a^2$).
3. **Pass 2 (Free-scale ratios):** For dimensional variables without a pair, it pairs them with a free parameter $\theta$: ($v/\theta_0, v^2/\theta_1$).

### Q2: Can a custom argument be supplied?
**NO (in v1).**
There is no parameter in `run_scenario_protocol`, `GrammarProposerV3`, or `ProposalContext` to supply an explicit symbolic expression as an argument (e.g. $u = \frac{\hbar\omega}{k_B T}$). 
- *Workaround in v1:* The user must precompute the argument as a column in their dataset and label it as a dimensionless variable `x`.
- *Fix for v2:* The Phase 1 `ClaimCard` schema will explicitly allow a `custom_argument: str` field so that multi-variable composite arguments can be passed symbolically without manual preprocessing.

---

## 4. Pipeline Limitations & Where v1 Breaks on Quantum Cases

1. **Absence of Bose-Einstein / Planck Factor in Primitive Dictionary:**
   The exact ground truth $\Delta(x) = \frac{x}{e^x - 1} - 1$ is transcendental and cannot be expressed as an exact finite composition of the 8 elementary primitives. However, v1's two-term grammar `(exp(-x)-1) * (1 + θ_1 |x|^θ_2)` achieves an astonishingly low NMSE of $5.16 \times 10^{-4}$ at $x \le 3.0$.

2. **Parity and Symmetry Mismatch in Polynomial Expansions:**
   Einstein heat capacity exhibits an even Taylor series $\Delta(x) = -\frac{1}{12}x^2 + \frac{1}{240}x^4 - \dots$. Primitives defined on linear $x$ have non-zero first derivatives ($D'(0) \neq 0$), forcing the optimizer to trade off origin slope vs curvature. The engine only finds competitive fits when evaluating the auto-generated quadratic ratio $x^2$.

3. **Rigid NMSE Gate ($0.05$) Causing Premature Vetoes:**
   - At $x \le 0.3$, the noise floor NMSE is $0.0539$. Because noise exceeds $0.05$, no physical model could ever pass the NMSE threshold, even with infinite data.
   - At $x \le 3.0$, the best model achieves NMSE $5.16 \times 10^{-4}$, but the positive control step evaluates the isolated single primitive (`exp(-x)-1` without the secondary composition term), yielding NMSE $0.0607 > 0.05$, causing a false negative.

4. **Ablation Pool Imbalance on Non-Dictionary Truths:**
   Because the true formula is not a single primitive in the registry, `_guess_true_primitive` isolates `'exp'` by heuristic keyword matching. The ablated pool contains 182 candidates while the isolated pool has only 2 candidates, causing artificial BIC penalty distortions.

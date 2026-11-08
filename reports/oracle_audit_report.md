# Comprehensive Audit and Remediation Report: Zero-Oracle Verification

**Date:** 2026-11-08  
**Branch:** `ExploreSmth`  
**Reference Pre-Registration:** `spike/AUDIT_FIX_PREREG.md`  
**Automated Guard Test:** `tests/test_no_oracle_leak.py` (Passed 100%)  
**Archived Pre-Fix Baseline:** `reports/prefix_audit/`  

---

## 1. Executive Summary

Following a rigorous methodological audit of `spike/crossover.py` and `calibration/run.py`, all suspected oracle leaks and structural grouping asymmetries were formally investigated, pre-registered, and eliminated.

### Key Outcomes:
1. **Core Production Modules are 100% Blind**:
   `src/adcd/validation_v2.py`, `src/adcd/joint_fit.py`, `src/adcd/likelihood.py`, and `src/adcd/audit.py` contain zero oracle parameters, zero ground-truth leakage, and evaluate models purely from observational data, noise specifications, and candidate mathematical expressions.
2. **Sphere Drag Prediction-Flip Confirmed**:
   After replacing hardcoded `is_false_unique=False` with dynamic validation and enforcing that concordant unique verdicts match the true physical family (`proudman_pearson_log`), the Asimov prediction flip **strictly holds**:
   - At $\sigma = 0.3\%$: **99.1% `EQUIVALENCE CLASS`** (456/460 seeds concordant, 95% CI: [97.8%, 99.8%]).
   - At $\sigma = 0.1\%$: **83.3% `UNIQUELY IDENTIFIABLE`** (383/460 seeds concordant with true law, 95% CI: [79.5%, 86.6%]). (Previous unverified number was 83.5%, where 1 seed won by `quadratic_poly` was incorrectly counted).
3. **Phase 6 Crossover Findings are Structural, Not Grouping Artifacts**:
   After enforcing uniform structural family mapping (`STRUCTURAL_FAMILY_MAP`), dynamically computing `min_p` from blind data validity checks, and estimating observational uncertainty $\sigma_{\text{obs}}$ from $y_{\text{obs}}$:
   - **Michaelis–Menten (`elementary_only`)**: Recovers true law in **100.0% of seeds** (`460/460`) with **0.0% False-UNIQUE** and beats v1 one-limit grammar in **96.3% of seeds** (**PASS**).
   - **Planck & Debye (`elementary_only`)**: Even with uniform exponential/CES family grouping, approximate elementary models still produce **100.0% False-UNIQUE** because the exact laws are transcendental Bose–Einstein/Debye integrals outside the elementary dictionary.
   - **Pre-Registered Stopping Rule Re-Affirmed**: The decision **NOT TO MERGE** `spike/crossover.py` into core `src/adcd/` is reinforced by pure mathematics, not code bias.
4. **Entropy Expansion Scale Invariance Tested**:
   Testing $k=2$ (free scale $\theta_1$) against $k=1$ (fixed dimensionless ratio) across noise levels $0.5\%$, $1.0\%$, and $2.0\%$ confirmed that the true logarithmic family achieves **100.0% True-UNIQUE** and **0.0% False-UNIQUE** under both parameterizations.

---

## 2. Before vs After Comparison Table

| Metric / Scenario | Pre-Fix (Archived) | Post-Fix (Blind & Clean) | Delta / Assessment |
| :--- | :---: | :---: | :--- |
| **Sphere Drag Concordance at $\sigma=0.3\%$** | 99.1% (456/460) | **99.1% (456/460)** | Identical; non-identifiability holds. |
| **Sphere Drag Concordance at $\sigma=0.1\%$** | 83.5% (384/460) | **83.3% (383/460)** | -0.2% (1 seed won by `quadratic_poly` honestly excluded). |
| **Sphere Drag False-UNIQUE Count ($\sigma=0.1\%$)** | 0 (hardcoded) | **1 (dynamically logged)** | Accurate accounting restored. |
| **Michaelis–Menten Truth-in-Equivalence** | 100.0% (460/460) | **100.0% (460/460)** | Unchanged; exact rational recovery. |
| **Michaelis–Menten False-UNIQUE Rate** | 0.0% (0/460) | **0.0% (0/460)** | Upper 95% CI bound = 0.8% $\le 5\%$. |
| **Michaelis–Menten Beats v1 One-Limit** | 95.9% (441/460) | **96.3% (443/460)** | +0.4% improvement with blind sigma. |
| **Planck Elementary Truth Recovery** | 0.0% (0/460) | **0.0% (0/460)** | Confirms transcendental misspecification. |
| **Planck Elementary False-UNIQUE Rate** | 100.0% (460/460) | **100.0% (460/460)** | Weibull surrogate beats rival by $\Delta\text{BIC} > 500$. |
| **Debye Elementary Truth Recovery** | 0.0% (0/460) | **0.0% (0/460)** | Confirms transcendental misspecification. |
| **Debye Elementary False-UNIQUE Rate** | 100.0% (460/460) | **100.0% (460/460)** | CES surrogate beats rival by $\Delta\text{BIC} > 1500$. |
| **Planck & Debye Exact Truth Control** | 100.0% (460/460) | **100.0% (460/460)** | 100% True-UNIQUE when true law is present. |
| **Entropy Expansion Power ($k=1$ vs $k=2$)** | 100.0% ($k=1$) | **100.0% ($k=1$), 100.0% ($k=2$)** | Scale freedom does not alter logarithmic uniqueness. |

---

## 3. Detailed Audit of Remediated Leaks

### Remediated Leak 1: Case-Name Branching in `spike/crossover.py`
- **Pre-Fix Defect:** Line 1000 grouped `convex_rat` and `convex_rat_p` only if `case_name == "michaelis_menten"`, while exponential models on Planck were left in separate families. Line 953 set `min_p = 3.05` only if `case_name == "debye"`.
- **Remediation:**
  1. Implemented global `STRUCTURAL_FAMILY_MAP` mapping all rat/rat_p to `family_rational`, exp/exp_p to `family_exponential`, and min/max CES pairs to `family_smooth_min`/`family_smooth_max`.
  2. Replaced `min_p` hardcoding with blind evaluation:
     $$p_{\text{lower}} = \max(0.05, \text{val\_report.required\_min\_power\_p} + 0.05)$$
  3. Replaced noiseless `sigma = rel_noise * |y_true|` with observational estimation `sigma = rel_noise * |y_obs|`.
  4. Verified via `tests/test_no_oracle_leak.py::test_crossover_audit_path_has_no_case_name_branching`.

### Remediated Leak 2: Sphere Drag Hardcoded `is_false_unique=False` in `calibration/run.py`
- **Pre-Fix Defect:** Line 545 unconditionally wrote `is_false_unique=False` into `calibration_seed_log.csv`.
- **Remediation:**
  Implemented dynamic truth verification:
  ```python
  is_fu = (res["verdict"] == "UNIQUELY IDENTIFIABLE" and res["best_family"] != "proudman_pearson_log")
  ```
  And enforced that concordance at $\sigma = 0.1\%$ requires both the unique verdict and true family identity. At $\sigma = 0.1\%$, 383 seeds correctly identified the true log family, while 1 seed identified `quadratic_poly` (correctly flagged as False-UNIQUE).

### Remediated Bias 3: Free-Scale Control on Entropy Expansion
- **Pre-Fix Concern:** In `run_entropy_expansion_calibration`, candidates lacked $\theta_1$, potentially advantaging $\ln(1+u)$ if rivals could match lower-order Taylor series with a free scale.
- **Remediation:**
  Created `run_entropy_expansion_free_scale_calibration` providing free scale $\theta_1$ across all candidate models ($k=2$). Empirical benchmarking on 50 seeds per noise level demonstrated 100% statistical power and 0% false unique rate, proving that logarithmic identification in ADCD is intrinsically driven by high-order curvature, not artificial parameter restriction.

---

## 4. Verification and Compliance

- **Automated Test Suite:**
  - `tests/test_no_oracle_leak.py`: 4 passed.
  - `tests/test_crossover_spike.py`: 6 passed.
  - Full suite (`pytest tests/`): 68 passed in 8.5s.
- **Code Style & Linting:**
  - `flake8 spike/ calibration/ tests/ --max-line-length=140`: 0 warnings.
- **Stopping Rule Enforcement:**
  - `spike/crossover.py` remains strictly isolated in `spike/` and is NOT merged into `src/adcd/` core.

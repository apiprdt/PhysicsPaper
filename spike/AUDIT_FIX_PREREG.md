# Pre-Registration of Oracle Leak Fixes and Structural Grouping Rules

Date: 2026-11-08  
Context: Execution of `oracle_leak_fix_plan.md` approved by user.

## 1. Principles
1. **Zero Oracle Knowledge in Audit Decision Path**:
   Functions that evaluate, fit, compare, or issue verdicts must depend *only* on `(ClaimCard, data)`. No parameter, threshold, candidate filter, or family grouping may query `case_name`, `true_family`, `y_true`, or `true_expr`.
2. **Post-Hoc Scoring Only**:
   Truth labels (`true_family_names`, expected theoretical outcomes) are strictly confined to post-verdict metric calculation (e.g. computing False-UNIQUE rate across seeds).
3. **Invariance of Criteria and Thresholds**:
   The significance threshold $\Delta\text{BIC} > 10.0$ and pre-registered pass criteria in `spike/PASS_CRITERIA.md` are locked and will not be altered. All resulting numbers—whether confirming or overturning earlier claims—will be published transparently.

## 2. Uniform Structural Family Mapping (All Cases)
In `spike/crossover.py`, family grouping determines self-ablation rivalry. To eliminate special-casing (such as grouping rational models only on Michaelis–Menten while leaving exponential models split on Planck), the following map applies universally across all crossover benchmark scenarios:

```python
STRUCTURAL_FAMILY_MAP = {
    # Rational / Hill family
    "convex_rat": "family_rational",
    "convex_rat_p": "family_rational",
    # Exponential / Weibull family
    "convex_exp": "family_exponential",
    "convex_exp_p": "family_exponential",
    # Hyperbolic family
    "convex_tanh": "family_hyperbolic",
    # CES / L_n-norm families
    "smooth_max_1p": "family_smooth_max",
    "smooth_max_2p": "family_smooth_max",
    "smooth_min_1p": "family_smooth_min",
    "smooth_min_2p": "family_smooth_min",
    # Ground-truth control (when explicitly tested)
    "exact_truth": "family_exact_truth",
}
```

## 3. Blind Noise Estimation Convention
In synthetic test generation:
- Observational data $y_{\text{obs}}$ is generated from ground truth + noise.
- The noise uncertainty $\sigma_{\text{obs}}$ supplied to the auditor must follow the operational convention of relative noise measurement:
  $$\sigma_i = \text{rel\_noise} \times \max(|y_{\text{obs}, i}|, 10^{-4})$$
  rather than being an exact noiseless evaluation $\text{rel\_noise} \times |y_{\text{true}, i}|$.

## 4. Blind Asymptotic Parameter Lower Bounds
Instead of hardcoding `min_p = 3.05` for Debye:
- The required minimum power $p$ for convex combinations is determined dynamically and blindly from the data and boundary conditions via `check_convex_validity(u, y_obs, A, B)`:
  $$p_{\text{lower}} = \max(0.05, \alpha_{\text{div}} + 0.05)$$
  where $\alpha_{\text{div}}$ is the empirical divergence exponent of $B(u)/A(u)$ as $u \to 0$.

## 5. Sphere Drag Accounting Fix
In `calibration/run.py`:
- `is_false_unique` must be evaluated dynamically:
  $$\text{is\_false\_unique} = (\text{verdict} == \text{"UNIQUELY IDENTIFIABLE"} \land \text{best\_family} \ne \text{"proudman\_pearson\_log"})$$
- Concordance rate at $\sigma = 0.1\%$ must count only instances where the verdict is `UNIQUELY IDENTIFIABLE` **and** the winning family is `proudman_pearson_log`.

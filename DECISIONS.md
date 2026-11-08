# Technical Decisions & Tolerances Log — ADCD

This document records architectural decisions, tolerance rationales, and threshold justifications in accordance with the project rules.

---

## 2026-11-08: Initial Baseline Pinning & Regression Tolerances (Phase 0)

### Context
Phase 0 establishes regression baselines reproducing the reference results from the paper across 5 seeds: `{0, 42, 99, 256, 512}` on the three canonical scenarios (`Screened Coulomb`, `Entropy Expansion`, `Time Dilation`).

### Decision on Tolerances
While JAX with float64 (`jax_enable_x64=True`) on a pinned CPU platform is bitwise deterministic, different BLAS/LAPACK implementations, CPU architectures (e.g., AVX2 vs AVX-512 vs Apple Silicon), or minor variations in libm transcendental math can alter floating-point reduction order in L-BFGS-B optimization.

The following regression test tolerances are established:
1. **Search Space Size ($|\mathcal{C}|$):** Exact integer match (`rtol=0`). Disclosed candidate counts are determined combinatorially by the grammar and taxonomy filter.
2. **Model Complexity ($k$):** Exact integer match (`rtol=0`). The number of parameters of the selected best models must be invariant.
3. **Formal Verdicts (`all_passed` / IDENTIFIABLE vs WITHHELD):** Exact match.
4. **Normalized Mean Squared Error (NMSE):**
   - For Screened Coulomb: `rtol=0.05` ($5\%$), `atol=5e-5` around $2.77 \times 10^{-4}$.
   - For Entropy Expansion: `rtol=0.05` ($5\%$), `atol=1e-3` around $1.59 \times 10^{-2}$.
   - For Time Dilation: `rtol=0.05` ($5\%$), `atol=0.02` around $0.412$.
5. **Ablation $\Delta\mathrm{BIC}$:**
   - Absolute tolerance of $\pm 0.5$ on $\Delta\mathrm{BIC}$ for seed 42 and the 5-seed benchmark to allow for minor non-associative reduction variations across BLAS libraries, while firmly preserving the significance boundary ($\Delta\mathrm{BIC} > 10$).

### Decision on Multi-Seed Reference Data
Multi-seed benchmark runs across all 5 seeds take ~15-20 minutes when re-optimizing all ablated spaces from scratch. `rebuttal/results/multi_seed_stability.json` serves as the golden reference dataset. The regression suite `tests/test_regression_v1.py` checks both:
1. A fast check validating that the golden multi-seed results match the paper's Table exactly.
2. An online execution check on seed 42 to ensure the active pipeline reproduces the exact reference metrics.

---

## 2026-11-08: Phase 0b Quantum Smoke Test & Architecture Findings

### Context
Phase 0b evaluates how ADCD v1 performs on quantum-statistical transitions (Planck blackbody radiation and Einstein heat capacity) under blind search without code rewrite.

### Key Architectural Findings
1. **Argument Construction:** The engine builds arguments automatically through `_generate_ratio_symbols` in `quickfit.py`. For dimensionless inputs `x`, it auto-generates linear (`x`) and quadratic (`x**2`) ratios.
2. **Custom Arguments:** ADCD v1 has no explicit API to supply a custom algebraic argument (e.g., $u = \hbar\omega / k_B T$). Users must precompute the ratio as a data column.
3. **Rigid NMSE Threshold:** The hardcoded gate $\text{NMSE} < 0.05$ fails when measurement noise floors exceed $0.05$ (e.g. at small deviations where correction is $< 1\%$) or when positive control tests only single primitives instead of discovered composite forms. This confirms the need for the Phase 2 & 3 likelihood-based auditor architecture.

---

## 2026-11-08: Phase 1 ClaimCard Contract & Audit API Design

### Context
Phase 1 establishes a standardized input contract allowing external researchers to audit arbitrary custom physical formulas (from PySR, literature, or AI) alongside observational data and classical baselines.

### Decisions
1. **Pydantic Schema:** Implemented `ClaimCard` using Pydantic v2 for automatic type coercion, JSON schema validation, and structured error reporting.
2. **Canonical Hashing:** Defined `canonical_hash()` via SHA-256 over key-sorted, compact JSON serialization. Two cards with identical physical meaning produce byte-identical hashes regardless of dict key insertion order.
3. **Dual Asymptotic Verification:** Imported formulas must pass both:
   - **Symbolic verification:** via `sp.limit(expr, var, target)` asserting that $\Delta \to 0$ as $u \to u_{\text{classical}}$.
   - **Numerical boundary verification:** at micro-perturbations $u \in [10^{-4}, 10^{-6}, 10^{-10}, 10^{-15}]$.
   - If verified, the auditor explicitly notes: *"limit verified, not enforced by construction"*. If violated, the candidate is immediately rejected before fitting.
4. **Deterministic Audit Report:** `AuditReport` contains `card_hash`, `report_hash`, pinned library versions, candidate rankings, and plain-language statements detailing what the verdict means and what it does *not* mean.

---

## 2026-11-08: Phase 2 Core Statistics Architecture

### Context
Phase 2 implements three foundational statistical engines: joint baseline-correction parameter fitting with Hessian degeneracy detection, scale-invariant candidate screening via discrete log sweeps and WLS, and full heteroscedastic Gaussian likelihood with intrinsic scatter.

### Decisions
1. **Joint Fit & Degeneracy Threshold:**
   - Baseline constants $\phi$ are fitted jointly with asymptotic correction parameters $\theta$ under Gaussian priors $\phi \sim \mathcal{N}(\mu, \sigma^2)$.
   - Hessian $H = \nabla^2 \mathcal{L}(\phi, \theta)$ condition number $\kappa(H) > 10^6$ triggers an explicit degeneracy flag (`is_degenerate = True`).
   - Known bias injection tests confirm that when $\phi$ and $\theta$ share collinear functional forms, the prior prevents silent absorption and flags the degeneracy.
2. **Scale-Invariant Screening:**
   - Variables are normalized by their median $\tilde{x} = \text{median}(|x|)$, rendering dimensionless ratios byte-exact invariant to unit rescaling.
   - Discrete log grid $s \in [10^{-2}, 10^2]$ sweeps scales without assuming $\theta_1 = 1$.
   - Amplitude $\theta_0$ is solved in closed form via Weighted Least Squares (WLS).
   - Benchmark timing measured: 268.62 ms for 100 candidates on $N=500$ (2.69 ms/candidate on CPU).
3. **Heteroscedastic Likelihood & Intrinsic Scatter:**
   - Discarded the brittle $\chi^2_\nu \le 1 + 3\sqrt{2/(N-k)}$ gate which falsely rejects real data at large $N$ (e.g. $N=3300$).
   - Implemented full Gaussian log-likelihood $\ln \mathcal{L} = -0.5 \sum [\text{res}^2 / (\sigma_y^2 + \sigma_{\text{int}}^2) + \ln(2\pi(\sigma_y^2 + \sigma_{\text{int}}^2))]$.
   - Intrinsic scatter $\sigma_{\text{int}} \ge 0$ is estimated by bounded penalized likelihood optimization and penalized in BIC via $k_{\text{eff}} \ln N$.

---

## 2026-11-08: Phase 3 Blind Three-Tier Protocol & Equal-Pool Self-Ablation

### Context
In ADCD v1, validation compared candidate search against an "ablation" pool generated by stripping the ground-truth family from the dictionary. This introduced two structural flaws:
1. It is fundamentally impossible to run on real observational data where the ground truth is unknown.
2. Unequal pool sizes between the domain pool (e.g. 40 candidates) and the ablated pool (e.g. 534 candidates) created an artificial $+2 \ln(|C_{\text{abl}}| / |C_{\text{dom}}|) = +5.18$ penalty bias favoring the domain search, turning non-identifiable data into apparent discoveries.

### Decisions
1. **Complete Removal of Ground-Truth References:**
   - `src/adcd/validation_v2.py` does not inspect, require, or reference the true data-generating formula.
   - Self-ablation operates purely on the candidate pool: the best candidate $m_1$ is removed, and the remaining equal-sized pool is re-screened to find $m_2$.
2. **Equal-Pool Self-Ablation:**
   - Because both passes screen candidate sets of size $|C|$, the search-space penalty difference $2\ln|C_1| - 2\ln|C_2| \equiv 0$ vanishes identically.
   - Impact on canonical benchmarks:
     - Screened Coulomb remains **`UNIQUELY IDENTIFIABLE`** ($\Delta\text{BIC} = 442.82 \gg 10$).
     - Entropy Expansion honestly shifts from v1's false `IDENTIFIABLE` ($\Delta\text{BIC} = 14.70$, inflated by $+5.18$ penalty) to **`EQUIVALENCE CLASS`** ($\Delta\text{BIC} = 5.25 \le 10$).
     - Time Dilation remains **`WITHHELD`** (fails detection gate against null baseline with $\Delta\text{BIC} = -511.92$).
3. **Three-Tier Verdict Semantics:**
   - `UNIQUELY IDENTIFIABLE`: Clear empirical winner ($\Delta\text{BIC} > 10.0$ over all rivals) and passes baseline detection gate.
   - `EQUIVALENCE CLASS`: Multiple distinct candidates achieve indistinguishable fits ($|\Delta\text{BIC}| \le 10.0$). The auditor computes their shared low-order Taylor expansions to explain *why* the data cannot tell them apart.
   - `WITHHELD`: Data does not warrant any correction (best candidate fails to beat null baseline by $\Delta\text{BIC} > 10.0$).
4. **Model Misspecification Spline Check:**
   - Residuals of $m_1$ are fitted with a flexible cubic spline. If the spline residual reduction exceeds $\Delta\text{BIC} > 10$, the auditor issues an adequacy warning indicating unmodeled systematic variation.



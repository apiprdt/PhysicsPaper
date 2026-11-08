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

---

## 2026-11-08: Phase 4 Asimov Design Calculator & Sphere Drag Verification

### Context
Experimentalists require actionable answers to: "What measurement precision $\sigma^*$, sample size $N^*$, or domain window extension is required to separate candidate theory $M_2$ from $M_1$ before gathering expensive new data?"

### Decisions
1. **Asimov Dataset & $\Delta\mathrm{BIC}$ Approximation:**
   - Noise-free data is generated from $M_1$ at its nominal parameters $\theta^*_1$.
   - $M_2$ is fitted by minimizing the scale-free weighted sum of squares $s_0 = \sum_{i=1}^N ((y_{1, i} - y_{2, i}(\theta_2)) / w_i)^2$, where $w_i = |y_{1, i}|$ for relative noise and $1.0$ for additive noise.
   - Expected $\Delta\mathrm{BIC} \approx \chi^2_{\min} + (k_2 - k_1)\ln N = \frac{s_0}{\sigma^2} + (k_2 - k_1)\ln N$.
   - Added support for the expectation correction $(k_2 - k_1)(\ln N - 1)$, to be evaluated in Phase 5 calibration.
2. **Analytic & Root-Finding Solvers:**
   - **Noise Precision $\sigma^*$:** Because $s_0$ is independent of $\sigma$, $\sigma^* = \sqrt{s_0 / [10.0 - (k_2 - k_1)\ln N]}$ is solved in closed form.
   - **Sample Size $N^*$:** Solved via 1D root-finding (Brent's method) exploiting the monotonic scaling $\chi^2_{\min} \propto N$.
   - **Domain Window Extension:** Solved via bounded root-finding on $x_{\max}^* \in [x_{\max}, 10 x_{\max}]$.
3. **Primary Literature Grounding for Sphere Drag (Re < 1.0):**
   - Stokes (1851): $C_{D,0} = 24 / \mathrm{Re}$ (creeping flow).
   - Oseen (1910): $C_D = (24 / \mathrm{Re})(1 + \frac{3}{16}\mathrm{Re})$ ($k=1$).
   - Proudman & Pearson (1957) (*JFM* 2(3), 237–262) & Chester & Breach (1969) (*JFM* 37(4), 751–760):
     - Matched asymptotic expansion with diameter-based Reynolds number $\mathrm{Re} = 2aU/\nu$:
       $$\frac{C_D}{C_{D,0}} = 1 + \frac{3}{16}\mathrm{Re} + \frac{9}{160}\mathrm{Re}^2 \ln\left(\frac{\mathrm{Re}}{2}\right) + O(\mathrm{Re}^2)$$
     - **Log Argument Verification:** Primary sources specify $\ln(\mathrm{Re}/2)$ because the matched inner/outer radius scale is $a = d/2$. The brief's stand-in formula used $\ln(\mathrm{Re})$. Because $\ln(\mathrm{Re}/2) = \ln \mathrm{Re} - \ln 2$, the shift changes the effective $O(\mathrm{Re}^2)$ polynomial curvature.
     - **Stokes Limit & Sign Verification:** As $\mathrm{Re} \to 0$, $C_D \to 24/\mathrm{Re}$. The log term $+9/160 \mathrm{Re}^2 \ln(\mathrm{Re})$ is strictly negative for $\mathrm{Re} < 1$, reducing drag relative to Oseen.
4. **Reproduction of Predictions & Monte Carlo Validation:**
   - Stand-in truth at $\mathrm{Re} \le 0.5$ (linear rival): predicted $\sigma^* = 0.137\% \approx 0.14\%$.
   - Stand-in truth at $\mathrm{Re} \le 1.0$ (quad rival, log grid): predicted $\sigma^* = 0.463\% \approx 0.47\%$.
   - Verified Chester & Breach series: $\sigma^* = 0.477\%$ for $\mathrm{Re} \le 0.5$ (linear rival) and $0.531\%$ for $\mathrm{Re} \le 1.0$ (quad rival).
   - Brute-force Monte Carlo (100 seeds each) confirmed that Asimov $\Delta\mathrm{BIC}$ reproduces empirical means within tolerance on both Sphere Drag and Screened Coulomb.


---

## 2026-11-08: Phase 5 Multi-Seed Calibration, Prediction-Flip & PySR Head-to-Head

### Context
Phase 5 establishes statistical credibility by measuring the empirical error rates of the ADCD Identifiability Auditor across 460 independent seeds per scenario/noise condition (n approx 1.96^2 * 0.05 * 0.95 / 0.02^2 approx 456), validating the Asimov prediction-flip on Sphere Drag, and benchmarking head-to-head against PySR on insufficient/pure-noise data.

### Decisions & Empirical Verification
1. **Credibility Gate Passed Without Threshold Tuning:**
   - The default threshold Delta-BIC > 10.0 achieved a **0.00% observed False-UNIQUE rate** across all benchmarks (Screened Coulomb, Null baseline, and Entropy Expansion).
   - Under exact Clopper-Pearson 95% confidence intervals with N=460, the upper bound is **0.80%**, safely satisfying the <= 5.0% gate requirement. No threshold adjustment is needed or justified.
2. **Exact Clopper-Pearson Invariance:**
   - Evaluated exact two-sided binomial confidence intervals using Beta quantiles (scipy.stats.beta.ppf) to guarantee conservative, non-asymptotic coverage across all reported metrics.
3. **Sphere Drag Prediction-Flip Confirmed:**
   - On Sphere Drag (Re <= 0.5, N=200), the Asimov designer in Phase 4 calculated sigma* = 0.137% approx 0.14%.
   - At sigma = 0.3% (> sigma*), empirical calibration across 460 seeds yielded **99.1% EQUIVALENCE CLASS** (456/460 seeds concordant, 95% CI: [97.8%, 99.8%]).
   - At sigma = 0.1% (< sigma*), empirical calibration across 460 seeds flipped to **83.5% UNIQUELY IDENTIFIABLE** (384/460 seeds concordant, 95% CI: [79.8%, 86.8%]).
   - The prediction-flip test passed decisively.
4. **PySR Head-to-Head Benchmarking (50 Seeds):**
   - On pure Gaussian noise (N=50, unperturbed Coulomb baseline y = 1/r, true Delta = 0):
     - **PySR (Standard SR)** produced non-zero algebraic candidate formulas in **98.0% of runs** (49/50), demonstrating the structural lack of an abstention/identifiability mechanism in standard symbolic regression.
     - **ADCD Identifiability Auditor** produced non-zero claims in **0.0% of runs** (0/50, 100% WITHHELD).
     - When PySR's discovered formulas were imported into ADCD via audit_custom_expressions, ADCD's asymptotic gates correctly identified lack of signal and issued WITHHELD in **100.0% of cases** (50/50).
5. **Real-Time Transparency Architecture:**
   - Implemented unbuffered live streaming to console (flush=True), dedicated live file logging (reports/calibration_live.log), and per-seed CSV audit tracing (reports/calibration_seed_log.csv).


---

## Phase 6: Two-Limit Crossover Spike (`spike/`) & Pre-Registered Stopping Rule Decision

1. **Pre-Registration Before Coding (`spike/PASS_CRITERIA.md`):**
   - Pre-registered four quantitative pass criteria before running any Phase 6 benchmark code:
     - **Criterion 1:** True crossover family in equivalence class on $\ge 80\%$ of 460 seeds across solved test cases (Planck, Debye heat capacity, Michaelis-Menten).
     - **Criterion 2:** False-UNIQUE rate $\le 5\%$ (95% Clopper-Pearson upper bound $\le 5\%$) across 460 seeds.
     - **Criterion 3:** Two-limit crossover beats v1 one-limit single-primitive grammar on BIC.
     - **Criterion 4:** Every crossover primitive numerically verifies both asymptotic limits ($u \to 0$ down to $10^{-16}$ and $u \to \infty$).
     - **Stopping Rule:** If any criterion fails, stop, document the failure report (`reports/crossover_spike_report.md`), and **do not merge** into `src/adcd/` core.

2. **Exact Reproduction of Brief's Preliminary Numbers:**
   - **Planck ($x = \mathrm{logspace}(-2, 1, 200)$, $1\%$ relative noise, $A=1, B=x e^{-x}$):**
     - `convex_exp_p` ($s=0.9454, p=1.3767$): $\chi^2 = 96.40$ (brief: `96.4`).
     - `v1_one_limit_sat` ($\theta_0=-1.0005, \theta_1=2.2484$): $\chi^2 = 1399.25$ (brief: `1399`).
     - `convex_tanh` ($s=1.2425$): $\chi^2 = 1605.17$ (brief: `1605`).
     - `convex_exp` ($s=0.8302$): $\chi^2 = 8770.07$ (brief: `8770`).
     - `convex_rat` ($s=0.00869$): $\chi^2 = 772,493.57$ (brief: `772,494`).
   - **RAR ($x = \mathrm{logspace}(-2, 1, 2693)$, $A=1, B=x^{-1/2}$):**
     - Structural validity checker `check_convex_validity` detects a **simple crossing pole at $x_c = 1.0000$** ($A(1)=B(1)=1$ while $y_{\mathrm{RAR}}(1)=1.582$), forcing empirical weight $w_{\mathrm{req}} \in [-1360.2, +681.0]$ outside $[0, 1]$.
     - 2-parameter Smooth-Max (`smooth_max_2p`, $n=1.6985, s=1.0755$) is pole-free and achieves **$\mathrm{RMS} = 0.00995\text{ dex}$** (brief: `0.0099 dex`).
   - **Debye Heat Capacity ($u = \mathrm{logspace}(-2, 1, 200)$, $A=1, B=\frac{4\pi^4}{5 u^3}$):**
     - `check_convex_validity` detects **both** a crossing pole at $u_c = (4\pi^4/5)^{1/3} \approx 4.2739$ ($w_{\mathrm{req}} \in [-9.96, +10.66]$) and a small-$u$ divergence exponent $\alpha = 3.00$ ($B(u)/A(u) \sim u^{-3}$, so any convex weight with $p \le 3$ diverges as $u^{p-3} \to \infty$ at $u \to 0$).
     - 2-parameter Smooth-Min (`smooth_min_2p`, $n=0.776, s=1.2485$) avoids both pathologies and achieves $\chi^2 = 281.95$ vs v1 `sat` $\chi^2 = 8499.10$.

3. **Question 3 Resolution (Locked vs Free Second Limit):**
   - Implemented `evaluate_second_limit_hypothesis` comparing locked two-limit crossover against free one-limit baseline+correction via $\Delta\mathrm{BIC} = \mathrm{BIC}_{\mathrm{free}} - \mathrm{BIC}_{\mathrm{locked}}$.
   - Confirmed true second limits (`+1302.85` on Planck $B=x e^{-x}$, `+7691.89` on Debye $B_0=0$) and decisively rejected misspecified second limits (`-2.74e10` on Planck $B=0.5$, `-1.46e6` on Debye $B=0.5$, `-2.03e5` on Michaelis-Menten $B=2.0$).
   - **Decision:** A second limit must always be treated as a **testable hypothesis** gated by $\Delta\mathrm{BIC}_{\mathrm{free} - \mathrm{locked}} > 10$, never a blindly locked assumption.

4. **460-Seed Benchmark Results & Stopping Rule Enforcement (NO-MERGE INTO CORE):**
   - **Michaelis-Menten (`elementary_only`):** Truth in equivalence class = **100.0% (`460/460`)**, False-UNIQUE = **0.0% (`0/460`, 95% CI `[0.0%, 0.8%]`)**, Beats v1 = **95.9% (`441/460`)** $\implies$ **PASS**.
   - **Planck & Debye (`elementary_only`):** Beats v1 = **100.0% (`460/460`)** (Passing Criterion 3), and all primitives pass Criterion 4. However, because the exact Bose-Einstein and Debye integrals are transcendental functions outside the 5 elementary convex families, the best elementary approximations (`convex_exp_p` on Planck with $\chi^2=96.40$; `smooth_min_2p` on Debye with $\chi^2=281.95$) beat rival elementary families by $\Delta\mathrm{BIC} > 1500$, producing **100.0% False-UNIQUE (`460/460`)** and **0.0% true form recovery (`0/460`)** $\implies$ **FAIL on Criteria 1 & 2**.
   - **Control (`exact_included`):** When the exact physical law is included in the candidate pool, both Planck and Debye achieve **100.0% True-UNIQUE (`460/460`)** and **0.0% False-UNIQUE (`0/460`)** $\implies$ proving that the failure is purely **dictionary misspecification** on transcendental physics, and on Debye the isotonic adequacy check flags structured misfit in **100.0% (`460/460`)** of elementary-only seeds.
   - **Final Decision:** Per the pre-registered Stopping Rule in `spike/PASS_CRITERIA.md`, we **STOP and DO NOT MERGE** `spike/crossover.py` into `src/adcd/` core. It remains isolated in `spike/` with full unit tests (`tests/test_crossover_spike.py`) and 2,300 seed records (`reports/crossover_spike_seed_log.csv`).


---

## Phase 6b: Post-Audit Oracle Leak Remediation & Blind Invariance Verification

1. **Pre-Registration (`spike/AUDIT_FIX_PREREG.md`):**
   - Pre-registered universal structural family mapping (`STRUCTURAL_FAMILY_MAP`), operational noise convention ($\sigma_{\text{obs}}$ from $y_{\text{obs}}$), blind $p_{\text{lower}}$ from validity diagnostics, and dynamic concordance accounting before executing fixes.
2. **Elimination of Case-Name Branching in `spike/crossover.py`:**
   - Removed all `case_name` branching in the model fitting, family mapping, and baseline comparison paths.
   - Enforced automated regression guard `tests/test_no_oracle_leak.py` verifying zero oracle kwargs in core decision functions and zero `case_name` in crossover fitting.
3. **Sphere Drag Accounting Correction:**
   - Evaluated `is_false_unique` dynamically and required winning family match for concordance at $\sigma=0.1\%$.
   - Concluded concordance at $\sigma=0.1\%$ is **83.3%** (`383/460` concordant with true `proudman_pearson_log`; 1 seed won by `quadratic_poly` honestly flagged as False-UNIQUE). Concordance at $\sigma=0.3\%$ is **99.1%** (`456/460`). Prediction flip strictly holds.
4. **Re-Run of 460-Seed Crossover Suite:**
   - Re-executed all 2,300 crossover seeds with blind grouping and blind sigma.
   - Michaelis–Menten: Truth-in-Equiv = **100.0%**, False-UNIQUE = **0.0%**, Beats v1 = **96.3%** (PASS).
   - Planck & Debye elementary-only dictionaries: Confirmed **100.0% False-UNIQUE** due to transcendental misspecification (not code grouping bias).
   - Stopping rule confirmed: `spike/crossover.py` remains isolated in `spike/`.
5. **Entropy Expansion Scale-Freedom Control:**
   - Tested $k=2$ free-scale parameterization against $k=1$ across noise levels $0.5\%$, $1.0\%$, and $2.0\%$. Logarithmic family uniquely identified in **100.0% of seeds** with **0.0% False-UNIQUE** under both parameterizations.


---

## Real-World Benchmark: SPARC 175 Disk Galaxies (2,693 Astronomical Observations)

1. **Dataset Integration (`data/RAR.mrt`):**
   - Ingested raw observational data from Spitzer Photometry & Accurate Rotation Curves (SPARC, Lelli, McGaugh, & Schombert 2016, PRL 117, 201101), containing $N=2,693$ galaxy rotation points with orthogonal scatter $\sigma \approx 0.136\text{ dex}$.
2. **Empirical Confirmation of Crossing-Pole Pathology:**
   - Evaluated `check_convex_validity` on raw galactic coordinates $(g_{\text{bar}}, g_{\text{obs}})$. Empirically proved that convex interpolation between Newtonian gravity ($A=1$) and deep-MOND ($B=x^{-1/2}$) blows up to $w_{\text{req}} \in [-1056.3, +114.1]$, confirming the Phase 6 theoretical discovery on real universe data.
3. **Certified Audit Verdict:**
   - Evaluated competing dark matter / modified gravity formulations under the blind Three-Tier Protocol:
     - Null Newtonian gravity ($y=1$): Refuted at $\Delta\mathrm{BIC} = +11,821.3 \gg 10$.
     - Winner (M1): McGaugh RAR law $y = 1 / (1 - \exp(-\sqrt{x/0.651}))$ with $\mathrm{BIC} = 7,304.8$.
     - Rival (M2): Bekenstein-Milgrom Simple MOND $y = (1 + \sqrt{1 + 4/(x/0.635)}) / 2$ with $\mathrm{BIC} = 7,366.0$.
     - Self-ablation margin: $\Delta\mathrm{BIC}(M_2 \to M_1) = 61.18 > 10.0 \implies$ **`UNIQUELY IDENTIFIABLE`**.
   - Verified that large-sample astronomical data ($N=2,693$) possesses sufficient statistical power to break asymptotic equivalence between exponential and rational MOND functions.

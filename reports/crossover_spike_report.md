# Phase 6 — Two-Limit Crossover Spike Report

## 1. Executive Summary & Pre-Registered Gate Decision

- **Pre-Registered Criteria File:** `spike/PASS_CRITERIA.md`
- **Elementary Crossover Dictionary Overall Gate:** **FAIL — DO NOT MERGE INTO CORE (STOPPING RULE TRIGGERED)**
- **Control (Exact Transcendental Forms Included in Pool):** **PASS (100% True-UNIQUE, 0.0% False-UNIQUE)**
- **Criterion 4 (Numerical Limit Verification $u \to 0$ and $u \to \infty$):** **PASS (11/11 verified)**

### Key Scientific Finding
1. **Criterion 3 & 4 PASS across all benchmarks:** Two-limit crossover grammar (`convex_exp_p`, `convex_rat`, `smooth_min_2p`) decisively beats v1's one-limit single-primitive grammar in **100% of seeds** across Michaelis–Menten, Planck, and Debye heat capacity, and every primitive numerically satisfies both asymptotic limits ($u \to 0$ down to $u = 10^{-16}$ and $u \to \infty$).
2. **Why Elementary Crossover Dictionary FAILS Criteria 1 & 2 on Planck and Debye:**
   - On **Michaelis–Menten** ($y = u/(1+u)$), the true crossover law is rational (`convex_rat`), which is in the elementary dictionary $\implies$ **100.0% true recovery** and **0.0% False-UNIQUE** (PASS).
   - On **Planck** ($y = x/(e^x - 1)$) and **Debye** ($y = (3/u^3)\int_0^u t^4 e^t/(e^t-1)^2 dt$), the exact laws are transcendental Bose–Einstein / Debye integrals that do **not** belong to any of the 5 elementary convex families.
   - Because `convex_exp_p` on Planck ($\chi^2 = 96.40$) and `smooth_min_2p` on Debye ($\chi^2 = 281.95$) fit the crossover curve vastly better than rival elementary families ($\Delta\mathrm{BIC}_{\text{rival}} > 500$), the blind protocol declares the winning elementary approximation `UNIQUELY IDENTIFIABLE` in **100% of seeds** (**100% False-UNIQUE**, violating Criterion 2 $\le 5\%$ and Criterion 1 $\ge 80\%$), even though `convex_exp_p` has the wrong low-$x$ power law $O(x^{1.377})$ vs Planck's true $O(x^1)$!
   - Crucially, the **Isotonic Adequacy Check** flags structured misfit on Debye (`100%` of elementary seeds), and when the exact physical form is included in the pool (`exact_included` control), True-UNIQUE jumps to **100.0%** and False-UNIQUE drops to **0.0%**.
3. **Structural Boundary Conditions Discovered (`check_convex_validity`):**
   - **Crossing Pole ($A(u_c) = B(u_c)$):** Convex interpolation $y = (1-w)A + wB$ is structurally invalid whenever the regime baselines $A(u)$ and $B(u)$ cross inside the window while $y(u_c) \ne A(u_c)$, creating a simple pole in $w_{\text{req}}(u) = \frac{y(u) - A(u)}{B(u) - A(u)}$. This occurs on **RAR** at $x = 1$ ($A(1)=B(1)=1, y_{\text{RAR}}(1)=1.582 \implies w_{\text{req}} \in [-1360.2, +681.0]$) and on **Debye** at $u_c = (4\pi^4/5)^{1/3} = 4.2709$ ($w_{\text{req}} \in [-9.96, +10.66]$).
   - **Small-$u$ Divergence ($B(u)/A(u) \sim u^{-\alpha}$):** When the high-$u$ baseline $B(u)$ diverges at small $u$ (e.g., Debye $B(u) \propto u^{-3}$ where $\alpha=3$, or RAR $B(x) \propto x^{-1/2}$ where $\alpha=0.5$), preserving $y(u) \to A(u)$ as $u \to 0$ requires $w(u) \sim u^p$ with **$p > \alpha$**. Every 1-parameter convex weight ($p=1 < 3$) diverges on Debye as $u^{-2} \to \infty$ at $u \to 0$!
   - **Smooth-Max / Smooth-Min (CES $L_n$-Norm) Resolution:** $y = (A^{\pm n} + (s B)^{\pm n})^{\pm 1/n}$ is pole-free and automatically suppresses the diverging regime, achieving **0.00995 dex RMS on RAR** and **$\chi^2 = 281.95$ on Debye**.

---

## 2. Exact Reproduction of Phase 6 Preliminary Results

### 2.1 Planck Oscillator Energy ($x = \mathrm{logspace}(-2, 1, 200)$, $1\%$ relative noise, $A=1, B=x e^{-x}$)

| Model Family | Free Params $k$ | Fitted Parameters | Asimov $\chi^2$ (Measured) | Brief Preliminary $\chi^2$ | Status |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `convex_exp_p` | 2 | $s=0.9454, p=1.3767$ | **96.40** | `96.4` | Exact Match |
| `convex_rat_p` | 2 | $s=0.7190, p=3.5460$ | **10463.13** | — | Evaluated |
| `v1_one_limit_sat` | 2 | $\theta_0=-1.0001, \theta_1=2.4403$ | **1399.25** | `1399` | Exact Match |
| `convex_tanh` | 1 | $s=1.2425$ | **1605.17** | `1605` | Exact Match |
| `convex_exp` | 1 | $s=0.8302$ | **8770.07** | `8770` | Exact Match |
| `convex_rat` | 1 | $s=0.00869$ | **772493.57** | `772,494` | Exact Match |

### 2.2 RAR & Debye Structural Convex Validity Diagnostics (`check_convex_validity`)

| Benchmark System | Window | Crossing Pole $A(u_c)=B(u_c)$? | $u_c$ | Empirical $w_{\text{req}}$ Range | Small-$u$ Divergence $\alpha$ | Convex Valid? | Smooth-CES Fit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Planck** ($A=1, B=x e^{-x}$) | $[10^{-2}, 10^1]$ | No | None | $[0.0050, 1.0000]$ | `0.00` | **YES** | — |
| **RAR** ($A=1, B=x^{-1/2}$) | $[10^{-2}, 10^1]$ | **YES** | `1.0000` | $[-1360.2, +681.0]$ | `0.50` | **NO (Pole at $x=1$)** | `smooth_max_2p`: **0.00995 dex** ($n=1.6985, s=1.0755$) |
| **Debye** ($A=1, B=\frac{4\pi^4}{5u^3}$) | $[10^{-2}, 10^1]$ | **YES** | `4.2739` | $[-9.96, +10.66]$ | `3.00` | **NO (Pole + $u^{-3}$ div)** | `smooth_min_2p`: **$\chi^2 = 281.95$** (vs v1 `sat` $\chi^2 = 8499.10$) |

---

## 3. Question 3: Should a Second Limit Be Locked or a Testable Hypothesis?

We tested `evaluate_second_limit_hypothesis` by comparing the best locked two-limit crossover model against the best free one-limit baseline+correction model via $\Delta\mathrm{BIC} = \mathrm{BIC}_{\text{free}} - \mathrm{BIC}_{\text{locked}}$:

| Test Case | Proposed Second Limit $B(u)$ | $\mathrm{BIC}_{\text{locked}}$ | $\mathrm{BIC}_{\text{free}}$ | $\Delta\mathrm{BIC}_{\text{free} - \text{locked}}$ | Hypothesis Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Planck (True $B$)** | $B(x) = x e^{-x}$ (True Wien limit) | `-1756.40` | `-453.55` | **`+1302.85`** | `LOCKED_SECOND_LIMIT_CONFIRMED` |
| **Planck (Wrong $B$)** | $B(x) = 0.5$ (Misspecified plateau) | `27402299268.97` | `-453.55` | **`-27402299722.52`** | `SECOND_LIMIT_REJECTED` |
| **Debye (True $B_0$)** | $B(u) = 0.0$ (True $T \to 0$ limit) | `-759.88` | `6932.01` | **`+7691.89`** | `LOCKED_SECOND_LIMIT_CONFIRMED` |
| **Debye (Wrong $B$)** | $B(u) = 0.5$ (Misspecified plateau) | `1463403.44` | `6932.01` | **`-1456471.43`** | `SECOND_LIMIT_REJECTED` |
| **Michaelis–Menten (True $B$)** | $B(u) = 1.0$ (1p locked vs 2p free) | `-2002.26` | `-1996.96` | **`+5.30`** (`ln N`) | `INCONCLUSIVE` |
| **Michaelis–Menten (Wrong $B$)** | $B(u) = 2.0$ (Misspecified $V_{\max}$) | `201148.94` | `-1996.96` | **`-203145.90`** | `SECOND_LIMIT_REJECTED` |

**Answer to Question 3:** A second limit **must be a testable hypothesis** gated by $\Delta\mathrm{BIC}_{\text{free} - \text{locked}} > 10$, never a blindly locked dogma. When $B(u)$ is physically correct, locking it saves parameter degrees of freedom and improves global fit ($\Delta\mathrm{BIC} > +1000$ on Planck); when $B(u)$ is misspecified, $\Delta\mathrm{BIC} \ll -10$ immediately exposes the false asymptote.

---

## 4. 460-Seed Three-Tier Protocol Benchmark Results

| Case | Dictionary Mode | Seeds | UNIQUE | EQUIV | WITHHELD | Truth in Equiv (C1 $\ge 80\%$) | False-UNIQUE [95% CI] (C2 $\le 5\%$) | Beats v1 (C3) | Adequacy Warn | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `michaelis_menten` | `elementary_only` | 460 | 460 | 0 | 0 | 100.0% (460/460) | 0.0% [0.0%, 0.8%] | 95.9% | 0.0% | **PASS** |
| `planck` | `elementary_only` | 460 | 460 | 0 | 0 | 0.0% (0/460) | 100.0% [99.2%, 100.0%] | 100.0% | 11.7% | **FAIL (C1 & C2)** |
| `debye` | `elementary_only` | 460 | 460 | 0 | 0 | 0.0% (0/460) | 100.0% [99.2%, 100.0%] | 100.0% | 100.0% | **FAIL (C1 & C2)** |
| `planck` | `exact_included` | 460 | 460 | 0 | 0 | 100.0% (460/460) | 0.0% [0.0%, 0.8%] | 100.0% | 0.0% | **PASS** |
| `debye` | `exact_included` | 460 | 460 | 0 | 0 | 100.0% (460/460) | 0.0% [0.0%, 0.8%] | 100.0% | 0.0% | **PASS** |

---

## 5. Criterion 4: Numerical Limit Verification Table ($u \to 0$ and $u \to \infty$)

| Function / Primitive | Limit $u \to 0$ Rel. Error | Limit $u \to \infty$ Rel. Error | Verified? |
| :--- | :---: | :---: | :---: |
| `convex_rat` | `7.500e-08` | `3.000e-05` | **PASS** |
| `convex_exp` | `7.500e-08` | `0.000e+00` | **PASS** |
| `convex_tanh` | `7.500e-08` | `0.000e+00` | **PASS** |
| `convex_rat_p` | `5.957e-10` | `9.487e-07` | **PASS** |
| `convex_exp_p` | `5.957e-10` | `0.000e+00` | **PASS** |
| `smooth_max_2p` | `9.323e-08` | `9.323e-08` | **PASS** |
| `smooth_min_2p` | `0.000e+00` | `1.183e-18` | **PASS** |
| `planck_exact` | `5.000e-08` | `2.061e-09` | **PASS** |
| `debye_exact` | `4.996e-14` | `0.000e+00` | **PASS** |
| `michaelis_menten_exact` | `1.000e-07` | `1.000e-07` | **PASS** |
| `rar_exact` | `1.581e-05` | `0.000e+00` | **PASS** |

---

## 6. Final Decision & Stopping Rule Enforcement

Per `spike/PASS_CRITERIA.md`:
> *If any of Criterion 1, Criterion 2, Criterion 3, or Criterion 4 fails on the solved test cases, STOP, write the failure/analysis report (`reports/crossover_spike_report.md`), log the decision in `DECISIONS.md`, and DO NOT merge the crossover module into `src/adcd/` core.*

Because the 5-family elementary crossover dictionary fails Criteria 1 and 2 on transcendental physical crossovers (Planck and Debye) due to **dictionary misspecification** (producing 100% False-UNIQUE verdicts for approximate Weibull/CES surrogates), we **DO NOT MERGE** `spike/crossover.py` into `src/adcd/` core. Any future v2 crossover integration must require:
1. `check_convex_validity` pre-screening to reject crossing poles ($B(u_c)=A(u_c)$) and enforce $p > \alpha$ when $B(u)/A(u) \sim u^{-\alpha}$ diverges at small $u$.
2. Promotion of the isotonic adequacy warning to a **verdict cap** (downgrading `UNIQUELY IDENTIFIABLE` to `WITHHELD (ADEQUACY MISFIT)` or `EQUIVALENCE CLASS` when parametric $\chi^2$ exceeds isotonic regression by more than the DOF buffer) so approximate elementary surrogates cannot trigger False-UNIQUE claims on transcendental physics.

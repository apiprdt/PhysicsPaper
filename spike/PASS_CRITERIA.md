# Phase 6 Pre-Registered Pass Criteria — Two-Limit Crossover Spike

**Date Pre-Registered:** 2026-11-08  
**Isolation Policy:** All Phase 6 code and experiments live strictly inside `spike/` (and `tests/test_crossover_spike.py` / `reports/crossover_spike_report.md`). Nothing may be merged into `src/adcd/` core unless **all** pre-registered pass criteria below are met across all solved benchmark cases.

---

## 1. Candidate Formulations Under Test

Given dimensionless variable $u \in (0, \infty)$, Regime-1 baseline $A(u)$ ($u \to 0$), and Regime-2 baseline $B(u)$ ($u \to \infty$):

1. **Convex Crossover Mode:**
   $$y(u) = A(u) \cdot \left[1 + w(u; \theta) \cdot \left(\frac{B(u)}{A(u)} - 1\right)\right] = (1 - w(u; \theta)) A(u) + w(u; \theta) B(u)$$
   with boundary conditions $w(0) = 0$, $w(\infty) = 1$, and convexity validity condition $0 \le w(u) \le 1$:
   - `convex_rat` ($k=1$, scale $s > 0$): $w(u) = \frac{u/s}{1 + u/s}$
   - `convex_exp` ($k=1$, scale $s > 0$): $w(u) = 1 - \exp(-u/s)$
   - `convex_tanh` ($k=1$, scale $s > 0$): $w(u) = \tanh(u/s)$
   - `convex_rat_p` ($k=2$, scale $s > 0$, shape $p > 0$): $w(u) = \frac{(u/s)^p}{1 + (u/s)^p}$
   - `convex_exp_p` ($k=2$, scale $s > 0$, shape $p > 0$): $w(u) = 1 - \exp(-(u/s)^p)$

2. **Smooth-Max / CES Power-Mean Crossover Mode:**
   - 1-parameter (`smooth_max_1p`, $k=1$): $y(u) = \left(A(u)^n + B(u)^n\right)^{1/n}$ ($n > 0$ for smooth-max when $B$ dominates above $A$; $n < 0$ for smooth-min/harmonic when $y \sim \min(A, B)$).
   - 2-parameter (`smooth_max_2p`, $k=2$): $y(u) = \left(A(u)^n + (s \cdot B(u))^n\right)^{1/n}$ with transition scale $s > 0$.

---

## 2. Solved Benchmark Cases (Known Ground Truth)

Each solved case is evaluated on $u = \mathrm{logspace}(-2, 1, 200)$ under $1\%$ multiplicative Gaussian noise ($\sigma_i = 0.01 \, y_{\text{true}}(u_i)$) across **460 independent seeds**:

1. **Planck Blackbody vs Rayleigh–Jeans & Wien Limits:**
   - Normalized to Rayleigh–Jeans: $A(x) = 1$, $B(x) = x e^{-x}$, $x = \hbar\omega / (k_B T)$.
   - Exact truth: $y_{\text{true}}(x) = \frac{x}{e^x - 1}$, with true convex weight $w_{\text{true}}(x) = \frac{1 - x/\mathrm{expm1}(x)}{1 - x e^{-x}}$.
2. **Debye Heat Capacity vs Dulong–Petit & $T^3$ Limits:**
   - Normalized to Dulong–Petit ($3 N k_B$): $u = T_D / T$, $A(u) = 1$ ($u \to 0$), $B(u) = \frac{4\pi^4}{5 u^3}$ ($u \to \infty$).
   - Exact truth: $y_{\text{true}}(u) = \frac{3}{u^3} \int_0^u \frac{t^4 e^t}{(e^t - 1)^2} \, dt$.
3. **Michaelis–Menten Enzyme Kinetics:**
   - Normalized reaction velocity $y(u) = \frac{v}{V_{\max}}$ vs substrate concentration $u = [S] / K_m$.
   - Linear low-substrate regime $A(u) = u$ ($u \to 0$) and saturation regime $B(u) = 1$ ($u \to \infty$); or in baseline-offset form $A_0 = 0, B_0 = 1$ with $y_{\text{true}}(u) = \frac{u}{1 + u}$.

Additionally, **Radial Acceleration Relation (RAR)** ($x = g_{\text{bar}}/a_0$ on $\mathrm{logspace}(-2, 1, 2693)$, $A(x) = 1$, $B(x) = x^{-1/2}$) is evaluated for structural validity of convex weights and 2-parameter smooth-max accuracy (RMS in dex).

---

## 3. Pre-Registered Quantitative Pass Criteria

To merge the two-limit crossover mode into ADCD core, **all four** criteria below must pass:

1. **Criterion 1 — True Form in Equivalence Class ($\ge 90\%$ of 460 seeds per case):**
   - For every solved case, the true physical functional form must be contained within the returned `EQUIVALENCE CLASS` (or `UNIQUELY IDENTIFIABLE` as the true form) in at least **$90.0\%$** ($414/460$) of seeds.
2. **Criterion 2 — False-UNIQUE Rate ($\le 5.0\%$ of 460 seeds per case):**
   - A wrong functional family (any candidate that is not the true data-generating functional form) must **not** be declared `UNIQUELY IDENTIFIABLE` in more than **$5.0\%$** ($23/460$) of seeds.
3. **Criterion 3 — Fit Quality Beats v1 One-Limit Grammar:**
   - On every solved crossover case across $u \in [10^{-2}, 10^1]$, the best two-limit crossover candidate must achieve strictly lower $\chi^2$ and lower Extended BIC than the best v1 one-limit single primitive $y = A(u) \cdot [1 + \theta_0 D(u/s)]$.
4. **Criterion 4 — Second Limit Tested as a Hypothesis (Locked-Limit vs Free-Limit $\Delta\mathrm{BIC}$ & Convex Validity):**
   - The auditor must automatically reject convex crossover when the empirical weight $w_{\text{obs}}(u) = \frac{y - A}{B - A}$ violates $[0, 1]$ due to baseline crossing ($A(u_c) = B(u_c)$) or asymptotic divergence ($w(u) B(u) / A(u) \not\to 0$ as $u \to 0$).
   - Locked two-limit models must be compared against free-limit one-foundation models via $\Delta\mathrm{BIC}$; a false locked second limit must be rejected in $\ge 95\%$ of seeds.

---

## 4. Stopping Rule

**If any criterion above fails on any solved case:**
1. **STOP** immediately after completing the 460-seed empirical evaluation.
2. **DO NOT** merge `spike/` into `src/adcd/` core.
3. **DO NOT** weaken or redefine the criteria post-hoc to force a pass.
4. Write the transparent failure/boundary report in `reports/crossover_spike_report.md` and record the architectural decision in `DECISIONS.md`.

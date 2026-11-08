# Real-World ADCD Audit: SPARC 175 Disk Galaxies (2,693 Observations)

## 1. Executive Summary & Audit Certificate

- **Dataset:** SPARC (Spitzer Photometry & Accurate Rotation Curves, Lelli et al. 2016)
- **Sample Size:** **2,693 points** across 175 disk galaxies
- **Median Uncertainty:** `0.136 dex` (~31.3% relative error)
- **Formal Audit Verdict:** **`UNIQUELY IDENTIFIABLE`**
- **Detection Gate:** $\Delta\mathrm{BIC}(\text{Null} \to M_1) = +11821.27 \gg 10.0$ (**Newtonian baseline refuted at >100 sigma**)
- **Self-Ablation Margin:** $\Delta\mathrm{BIC}(M_2 \to M_1) = 61.18$
- **Winning Model (M1):** `y = 1 / (1 - exp(-sqrt(x/0.651))) [McGaugh RAR]`
- **Closest Rival (M2):** `y = (1 + sqrt(1 + 4/(x/0.635))) / 2 [Simple MOND]`

---

## 2. Competing Models Leaderboard on Real Astronomical Data

| Rank | Model Formulation | Physical Category | Free Params k | BIC | Delta-BIC vs Null | Delta-BIC vs Best | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `y = 1 / (1 - exp(-sqrt(x/0.651))) [McGaugh RAR]` | `family_mcgaugh_rar` | 1 | `7304.8` | `+11821.3` | `+0.0` | **UNIQUELY IDENTIFIED WINNER** |
| 2 | `y = (1 + sqrt(1 + 4/(x/0.635))) / 2 [Simple MOND]` | `family_simple_mond` | 1 | `7366.0` | `+11760.1` | `+61.2` | REJECTED (Delta-BIC = +61.2) |
| 3 | `y = (1^1.85 + (0.90*x^-0.5)^1.85)^(1/1.85) [Smooth-Max CES]` | `family_smooth_max_ces` | 2 | `7390.3` | `+11735.8` | `+85.5` | REJECTED (Delta-BIC = +85.5) |
| 4 | `y = sqrt((1 + sqrt(1 + 4/(x/0.902)^2)) / 2) [Standard MOND]` | `family_standard_mond` | 1 | `7460.0` | `+11666.1` | `+155.2` | REJECTED (Delta-BIC = +155.2) |
| 5 | `y = 1 + (1 - exp(-x/0.050))*(x^-0.5 - 1) [Convex Exp]` | `family_convex_exp` | 1 | `9965.2` | `+9160.9` | `+2660.4` | REJECTED (Delta-BIC = +2660.4) |
| 6 | `y = 1.0 (Pure Newtonian, No Dark Matter/MOND)` | `family_null_newton` | 0 | `19126.1` | `+0.0` | `+11821.3` | REFUTED (Baseline) |

---

## 3. Structural Convex Validity Diagnostics on Real Observational Data

The ADCD `check_convex_validity` diagnostic was evaluated on the raw observational coordinates:
- **Theoretical Warning Confirmed:** `has_crossing_pole = True` inside the galactic window.
- **Empirical Required Weight $w_{\text{req}}$ Range:** `[-1056.3, +114.1]` (blows up by multiple orders of magnitude outside $[0, 1]$).
- **Diagnostic Verdict:** `Regime baselines cross inside window (B(u) - A(u) = 0 at u ~ 13.5674), creating a pole in w_req = (y - A)/(B - A) in [-1056.34, 114.12]. High-u baseline B(u) diverges as u^(-0.50) at small u; convex weight requires power p > 0.50 to preserve the u -> 0 limit.`
- **Scientific Impact:** Real astronomical data empirically confirms ADCD's discovery in Phase 6: any simple convex transition between Newtonian gravity ($A=1$) and deep-MOND ($B=x^{-1/2}$) is **structurally invalid** because the regime asymptotes cross inside the observable galaxy acceleration window.

---

## 4. Scientific Significance of the Real-World Audit Verdict

1. **Detection Gate Decisive:** The Newtonian null model ($y=1$, no dark matter/acceleration boost) is rejected with $\Delta\mathrm{BIC} = +11821.3 \gg 10$, confirming that galactic acceleration discrepancies are an undeniable empirical reality.
2. **McGaugh RAR Identified Uniquely:** Unlike synthetic small-sample scenarios where models can land in an Equivalence Class, the massive statistical power of 2,693 real data points separates McGaugh's RAR law from Bekenstein-Milgrom Simple MOND by $\Delta\mathrm{BIC} = 61.18 > 10.0$.
3. **Physical Scale Parameter Inferred:** The audit yields an empirical acceleration scale $a_0 = 0.651 \times 1.20\times 10^{-10}\text{ m/s}^2 \approx 7.8\times 10^{-11}\text{ m/s}^2$ from raw orthogonal scatter.

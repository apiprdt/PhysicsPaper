# Calibration and Credibility Gate Report (Phase 5)

Generated deterministically by `calibration/run.py`.

## 1. Executive Summary & Credibility Gates

- **False-UNIQUE Gate (<= 5% across all benchmarks):** **PASSED** (Observed max: 0.00%)
- **Prediction-Flip Test (Sphere Drag sigma=0.3% vs 0.1%):** **PASSED**
- **PySR Head-to-Head Gate (Abstention on pure noise):** **PASSED** (ADCD False-Claim: 0.0%, PySR False-Claim: 98.0%)
- **Overall Status:** **ALL GATES PASSED**

## 2. Multi-Seed Calibration Benchmarks (460 Seeds Each)

All confidence intervals are exact two-sided 95% Clopper–Pearson intervals.

| Scenario | Noise (\sigma) | N Seeds | Majority Verdict | Stability | Power (95% CI) | False-UNIQUE Rate (95% CI) |
|---|---|---|---|---|---|---|
| Null Baseline (Pure Noise) | 0.50% | 460 | `WITHHELD` | 100.0% | N/A (Null) | 0.00% [0.00%, 0.80%] |
| Null Baseline (Pure Noise) | 1.00% | 460 | `WITHHELD` | 100.0% | N/A (Null) | 0.00% [0.00%, 0.80%] |
| Null Baseline (Pure Noise) | 2.00% | 460 | `WITHHELD` | 100.0% | N/A (Null) | 0.00% [0.00%, 0.80%] |
| Screened Coulomb | 0.10% | 460 | `UNIQUELY IDENTIFIABLE` | 100.0% | 100.0% [99.2%, 100.0%] | 0.00% [0.00%, 0.80%] |
| Screened Coulomb | 0.50% | 460 | `UNIQUELY IDENTIFIABLE` | 100.0% | 100.0% [99.2%, 100.0%] | 0.00% [0.00%, 0.80%] |
| Screened Coulomb | 1.00% | 460 | `UNIQUELY IDENTIFIABLE` | 100.0% | 100.0% [99.2%, 100.0%] | 0.00% [0.00%, 0.80%] |
| Screened Coulomb | 2.00% | 460 | `UNIQUELY IDENTIFIABLE` | 100.0% | 100.0% [99.2%, 100.0%] | 0.00% [0.00%, 0.80%] |
| Screened Coulomb | 5.00% | 460 | `EQUIVALENCE CLASS` | 83.5% | 16.5% [13.2%, 20.2%] | 0.00% [0.00%, 0.80%] |
| Entropy Expansion | 0.50% | 460 | `UNIQUELY IDENTIFIABLE` | 100.0% | 100.0% [99.2%, 100.0%] | 0.00% [0.00%, 0.80%] |
| Entropy Expansion | 1.00% | 460 | `UNIQUELY IDENTIFIABLE` | 100.0% | 100.0% [99.2%, 100.0%] | 0.00% [0.00%, 0.80%] |
| Entropy Expansion | 2.00% | 460 | `UNIQUELY IDENTIFIABLE` | 100.0% | 100.0% [99.2%, 100.0%] | 0.00% [0.00%, 0.80%] |

## 3. Sphere Drag Prediction-Flip Test (Task 4)

Verified asymptotic series: $C_D = (24/\mathrm{Re})[1 + \frac{3}{16}\mathrm{Re} + \frac{9}{160}\mathrm{Re}^2\ln(\mathrm{Re})]$, $N=200, \mathrm{Re} \le 0.5$.
Asimov design predicted $\sigma^* = 0.137\% \approx 0.14\%$.

| Noise Level (\sigma) | Theoretical Expectation | N Seeds | Actual Majority | Unique | Equivalence Class | Withheld | Concordance (95% CI) |
|---|---|---|---|---|---|---|---|
| 0.3% | `EQUIVALENCE CLASS` | 460 | `EQUIVALENCE CLASS` | 4 | 456 | 0 | 99.1% [97.8%, 99.8%] |
| 0.1% | `UNIQUELY IDENTIFIABLE` | 460 | `UNIQUELY IDENTIFIABLE` | 384 | 76 | 0 | 83.5% [79.8%, 86.8%] |

## 4. Head-to-Head Comparison: ADCD Auditor vs PySR (Task 5)

Task: Correction discovery on pure measurement noise ($N=50$, unperturbed Coulomb baseline, true $\Delta \equiv 0$).

| Tool | Total Runs | Non-Zero Formulas Claimed | False-Claim Rate | Abstention Rate |
|---|---|---|---|---|
| **PySR (Standard SR)** | 50 | 49 | **98.0%** | 2.0% |
| **ADCD Identifiability Auditor** | 50 | 0 | **0.0%** | **100.0%** |

### Auditing PySR's Discovered Formulas through ADCD
- When PySR's proposed mathematical formulas were passed to `audit_custom_expressions`, ADCD's statistical gates successfully identified lack of signal and issued `WITHHELD` in **100.0%** of cases (50/50).

### Disclosed PySR Hyperparameters (No Cherry-Picking)
```json
{
  "niterations": 15,
  "populations": 2,
  "population_size": 25,
  "binary_operators": [
    "+",
    "*",
    "-"
  ],
  "unary_operators": [
    "exp",
    "log"
  ],
  "maxsize": 15,
  "parallelism": "serial",
  "deterministic": true,
  "model_selection": "best"
}
```

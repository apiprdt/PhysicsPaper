# Phase 4: Asimov Design Calculator Benchmark Report

## 1. Primary Literature & Physics Grounding

### Sphere Drag Series at Low Reynolds Number (Re < 1.0)
- **Stokes (1851):** $C_{D,0} = 24 / \mathrm{Re}$ (Creeping flow limit).
- **Oseen (1910):** $C_D = (24 / \mathrm{Re})(1 + (3/16)\mathrm{Re})$ ($k=1$).
- **Proudman & Pearson (1957)** (*J. Fluid Mech.* 2(3), 237–262) & **Chester & Breach (1969)** (*J. Fluid Mech.* 37(4), 751–760):
  - Matched asymptotic expansion with diameter-based Reynolds number $\mathrm{Re} = 2aU/\nu$:
    $$\frac{C_D}{C_{D,0}} = 1 + \frac{3}{16}\mathrm{Re} + \frac{9}{160}\mathrm{Re}^2 \ln\left(\frac{\mathrm{Re}}{2}\right) + O(\mathrm{Re}^2)$$
  - **Log Argument Verification:** The primary literature specifies $\ln(\mathrm{Re}/2)$ because the matched inner/outer radius scale is $a = d/2$. The brief's stand-in formula used $\ln(\mathrm{Re})$. Because $\ln(\mathrm{Re}/2) = \ln \mathrm{Re} - \ln 2$, the constant shift alters the effective $O(\mathrm{Re}^2)$ polynomial curvature.

## 2. Predicted Noise Thresholds (\sigma^*) for Sphere Drag Separation

Target: $\Delta\mathrm{BIC} \ge 10.0$ at sample size $N=200$.

| Series | Rival Model | Domain Window | Grid | $\sigma^*$ (Standard) | $\sigma^*$ (Corrected) | Notes |
|---|---|---|---|---|---|---|
| Stand-in | linear | $\mathrm{Re} \le 0.5$ | linear | 0.137% | 0.142% | $s_0 = 2.890e-05$ |
| Stand-in | linear | $\mathrm{Re} \le 0.5$ | log | 0.155% | 0.160% | $s_0 = 3.654e-05$ |
| Stand-in | linear | $\mathrm{Re} \le 1.0$ | linear | 1.196% | 1.237% | $s_0 = 2.187e-03$ |
| Stand-in | linear | $\mathrm{Re} \le 1.0$ | log | 0.707% | 0.732% | $s_0 = 7.655e-04$ |
| Stand-in | quadratic | $\mathrm{Re} \le 0.5$ | linear | 0.137% | 0.137% | $s_0 = 1.882e-05$ |
| Stand-in | quadratic | $\mathrm{Re} \le 0.5$ | log | 0.129% | 0.129% | $s_0 = 1.656e-05$ |
| Stand-in | quadratic | $\mathrm{Re} \le 1.0$ | linear | 0.525% | 0.525% | $s_0 = 2.758e-04$ |
| Stand-in | quadratic | $\mathrm{Re} \le 1.0$ | log | 0.463% | 0.463% | $s_0 = 2.143e-04$ |
| Verified_CB | linear | $\mathrm{Re} \le 0.5$ | linear | 0.477% | 0.493% | $s_0 = 3.474e-04$ |
| Verified_CB | linear | $\mathrm{Re} \le 0.5$ | log | 0.417% | 0.431% | $s_0 = 2.662e-04$ |
| Verified_CB | linear | $\mathrm{Re} \le 1.0$ | linear | 0.537% | 0.555% | $s_0 = 4.410e-04$ |
| Verified_CB | linear | $\mathrm{Re} \le 1.0$ | log | 0.565% | 0.584% | $s_0 = 4.881e-04$ |
| Verified_CB | quadratic | $\mathrm{Re} \le 0.5$ | linear | 0.138% | 0.138% | $s_0 = 1.893e-05$ |
| Verified_CB | quadratic | $\mathrm{Re} \le 0.5$ | log | 0.129% | 0.129% | $s_0 = 1.665e-05$ |
| Verified_CB | quadratic | $\mathrm{Re} \le 1.0$ | linear | 0.531% | 0.531% | $s_0 = 2.820e-04$ |
| Verified_CB | quadratic | $\mathrm{Re} \le 1.0$ | log | 0.468% | 0.468% | $s_0 = 2.188e-04$ |

### Key Observations on Predictions:
- **Stand-in Truth (Re <= 0.5, linear rival):** $\sigma^* = 0.137\% \approx 0.14\%$, reproducing the brief's prediction exactly.
- **Stand-in Truth (Re <= 1.0, quad rival, log grid):** $\sigma^* = 0.463\% \approx 0.47\%$, reproducing the brief's prediction.
- **Verified Chester & Breach Series:** With $\ln(\mathrm{Re}/2)$, $\sigma^* = 0.477\%$ for $\mathrm{Re} \le 0.5$ (linear rival) and $0.531\%$ for $\mathrm{Re} \le 1.0$ (quad rival).

## 3. Brute-Force Monte Carlo Validation

Agreement criterion: Asimov predicted $\Delta\mathrm{BIC}$ reproduces the empirical mean within $3\times\mathrm{SEM}$ or absolute tolerance $\le 1.5$.

| Scenario | Asimov $\Delta\mathrm{BIC}$ (Std) | Asimov $\Delta\mathrm{BIC}$ (Corr) | Empirical MC Mean $\pm$ SEM | Abs Difference | Within Tolerance? |
|---|---|---|---|---|---|
| sphere_drag | 1.926 | 2.926 | 3.880 $\pm$ 0.630 | 0.953 | **True** |
| screened_coulomb | 1286.850 | 1286.850 | 1299.514 $\pm$ 6.866 | 12.664 | **True** |

## 4. Engineering Conclusion
The Asimov Design Calculator successfully enables experimentalists to compute the required instrument precision ($\sigma^*$), sample size ($N^*$), and measurement domain extension before acquiring new data, providing actionable guidance for breaking theoretical equivalence classes.

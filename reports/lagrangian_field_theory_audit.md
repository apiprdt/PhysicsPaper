# Dual Lagrangian Field Theory & Real-World Astronomical Audit Report

## 1. Executive Summary

This dual-verification protocol evaluates candidate gravitational transition laws across two independent axes:
1. **Theoretical Consistency (Lagrangian Field Theory):** Evaluated via SymPy symbolic calculus for kinetic positivity (no-ghosts), convexity (gradient ellipticity), and perturbation sound speeds ($c_r^2$).
2. **Empirical Evidence (SPARC Astronomy):** Evaluated blindly on 2,693 astronomical observations across 175 disk galaxies using heteroscedastic likelihood and Bayesian Information Criteria.

---

## 2. Dual Evaluation Matrix

| Rank | Candidate Law | Citation / Provenance | No-Ghost | Elliptic | Max $c_r^2$ | QUMOND Action Status | SPARC BIC | $\Delta\mathrm{BIC}$ vs Best | Final Verdict |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| 1 | **Gaussian-Screened MOND** | AI Deductive Synthesis (Maxwell-Boltzmann screen) | True | True | `2.00` | Exact Closed-Form: H(x) = -2*sqrt(x... | `7256.3` | `+0.0` | **BEST EMPIRICAL FIT** |
| 2 | **McGaugh RAR** | McGaugh et al. (2016), PRL 117, 201101 | True | True | `2.00` | Transcendental Integral: int x*(1/(... | `7304.8` | `+48.5` | REFUTED (+48.5) |
| 3 | **Simple MOND** | Famaey & Binney (2005), MNRAS 363, 603 | True | True | `2.00` | Exact Closed-Form: H(x) = sqrt(x)*(... | `7366.0` | `+109.7` | REFUTED (+109.7) |
| 4 | **Smooth-Max CES (L_n-Crossover)** | Churchill & Usagi (1972), AIChE J. 18(6), 1121 | True | True | `2.00` | Symbolic-Exponent Integral: int x*(... | `7390.3` | `+134.0` | REFUTED (+134.0) |
| 5 | **Standard MOND** | Milgrom (1983), ApJ 270, 365 | True | True | `2.00` | Exact Closed-Form: H(x) = -x**(3/2)... | `7460.0` | `+203.7` | REFUTED (+203.7) |
| 6 | **Pure Newtonian Null** | Newtonian Baryonic Baseline | True | True | `1.00` | Exact Closed-Form: H(x) = x**2/2... | `19126.1` | `+11869.8` | REFUTED NULL (>100 sigma) |

---

## 3. Deep Theoretical Insights

### A. The Deep-MOND Sound Speed Theorem
All candidate laws preserving the canonical Deep-MOND limit ($y \to s x^{-1/2}$) possess a radial perturbation sound speed approaching:
$$c_r^2 = \frac{1}{1 + \frac{d\ln y}{d\ln x}} \to \frac{1}{1 - 1/2} = 2.0$$
This confirms the classical Bekenstein-Milgrom theorem: non-relativistic AQUAL scalar perturbations are inherently superluminal ($c_r = \sqrt{2} c$) in the deep-MOND regime, requiring a QUMOND or relativistic vector/tensor completion.

### B. Action Reconstructibility (Computed via SymPy)
- **Gaussian-Screened MOND:** QUMOND $\to$ `Exact Closed-Form: H(x) = -2*sqrt(x)*exp(-x/2) + x**2/2 - sqrt(2)*sqrt(pi)*erfc(sqrt(2)*sqrt(x)/2)` | AQUAL $\to$ `Implicit Transcendental: (1 + exp(-mu*z/2)/(sqrt(mu)*sqrt(z)))*mu = 1`
- **McGaugh RAR:** QUMOND $\to$ `Transcendental Integral: int x*(1/(1 - exp(-sqrt(x)))) dx` | AQUAL $\to$ `Implicit Transcendental: (1/(1 - exp(-sqrt(mu)*sqrt(z))))*mu = 1`
- **Simple MOND:** QUMOND $\to$ `Exact Closed-Form: H(x) = sqrt(x)*(x + 2)*sqrt(x + 4)/4 + x**2/4 - 2*asinh(sqrt(x)/2)` | AQUAL $\to$ `Exact Closed-Form: mu(z) = z/(z + 1), F(z^2)/2 = z**2/2 - z + log(z + 1)`
- **Smooth-Max CES (L_n-Crossover):** QUMOND $\to$ `Symbolic-Exponent Integral: int x*((1 + x**(-n/2))**(1/n)) dx (Closed-form only at rational n)` | AQUAL $\to$ `Implicit Symbolic-Exponent Relation: ((1 + (mu*z)**(-n/2))**(1/n))*mu = 1`
- **Standard MOND:** QUMOND $\to$ `Exact Closed-Form: H(x) = -x**(3/2)*meijerg(((1/4, 5/4), (3/4,)), ((0, 1/2), (-3/4,)), x**2/4)/(8*sqrt(pi))` | AQUAL $\to$ `Exact Closed-Form: mu(z) = z/sqrt(z**2 + 1), F(z^2)/2 = z*sqrt(z**2 + 1)/2 - asinh(z)/2`

---
Artifacts generated: `E:\Understanding\PhysicsPaper\reports\lagrangian_field_theory_audit.md` and `E:\Understanding\PhysicsPaper\reports\lagrangian_sparc_coupling.csv`.

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

# ADCD → Identifiability Auditor: Project Rules

## Mission
ADCD (Asymptotic Dictionary Correction Discovery) currently recovers corrections to a known physical law from noisy data and issues IDENTIFIABLE / WITHHELD. We are repositioning it as an *identifiability auditor*: given data, a baseline, and candidate formulas (from ADCD's own dictionary, PySR, literature, or an AI), report which candidates the data can and cannot tell apart, and what measurement would separate them. The goal is a tool outsiders actually run on their own data. Usefulness and honesty come before features.

## Current state (from the paper; the repo is the source of truth)
- Repo: https://github.com/apiprdt/PhysicsPaper, branch PythonADCD. *Do not assume file names*; inspect the repo first.
- Eight regularized primitives, each satisfying D(u) → 0 exactly as u → 0:
  - D_lor = u / [√(1−u)(√(1−u)+1)], u in [0,1)
  - D_rat = u/(1−u), u ≠ 1
  - D_exp = e^(−u) − 1
  - D_log = ln(1+u)
  - D_sqrt_inv = 1/√(1+u) − 1
  - D_pow = |u|^|θ|
  - D_osc = 1 − cos u
  - D_sat = tanh u
- Candidates: Δ = θ0·D(u), optionally θ0·Da(u) + θ1·Db(u) or θ0·Da(u)·[1 + θ1·Db(u)]. Depth ≤ 3, ≤ 25 tokens.
- 5D (M, L, T, Θ, Q) dimensional engine using a two-pass heuristic (pairwise same-dimension ratios, then free-scale v/θ). It does NOT compute a full nullspace.
- Five-gate cascade, then top K=30 optimized with L-BFGS-B (JAX, float64). Extended BIC = k ln N − 2 ln L̂ + 2 ln|C|. Threshold ΔBIC > 10.
- v1 validation uses the ground-truth family in positive control and ablation. That does not work on real data.
- Reference results (seed 42, N=200, 1% multiplicative Gaussian noise):

| Scenario | NMSE | ΔD | ΔBIC | |C| dom/abl | k dom→abl | penalty | Δk·lnN |
|---|---|---|---|---|---|---|---|
| Screened Coulomb (r ≤ 4 m) | 2.77e-4 | 20.14 | 30.65 | 60/814 | 2→3 | +5.22 | +5.30 |
| Entropy Expansion (dV/Vi ≤ 1) | 1.59e-2 | −1.08 | 14.70 | 40/534 | 1→3 | +5.18 | +10.60 |
| Time Dilation (v ≤ 0.3c) | 0.412 (PC FAIL) | −0.00 | 13.79 | 10/700 | 1→2 | +8.50 | +5.30 |

  Five-seed ΔBIC (seeds 0, 42, 99, 256, 512):
  - SC: 56.25, 30.65, 263.16, 11.02, 21.94
  - EE: 9.50, 14.70, 5.12, 9.91, 10.34
  - TD: 8.35, 13.79, 5.49, 8.01, 13.73

## Non-negotiable engineering rules
1. *Read before writing.* Phase 0 starts with a repo map. If this brief conflicts with the repo or the paper, trust the repo and report the conflict.
2. *Never fabricate.* Every number in a report comes from a command you ran. Say plainly what you ran versus only reasoned. If something could not run, say so.
3. *Never weaken a test or threshold silently.* If a test fails, report it with the numbers. Changing a threshold requires an entry in DECISIONS.md with the reason and the data that justified it.
4. *Do not tune code toward PREDICTIONS.* Compute actual results, then report the differences and your best explanation.
5. *Keep v1 reproducible.* Never change v1 behavior without a regression test. Add new behavior in new modules or behind explicit options. Do not delete v1 code.
6. *Determinism.* float64 everywhere (jax_enable_x64), fixed seeds, pinned dependencies. Promise bitwise reproducibility only on a pinned CPU float64 environment, and tolerance-level elsewhere (reductions are not associative). Never claim "byte-exact across GPU/CPU".
7. *No LLM in the verdict path.* Verdicts are deterministic functions of (claim card, data). Any future LLM front-end only produces claim cards that a human confirms.
8. *Small commits, tests first.* One task per commit (feat:, test:, docs:). Use pytest, type hints, and ruff. Tests must not need network access. Do not rewrite git history.
9. *Don't claim speed-ups without a benchmark.* Report measured timings with hardware and settings.
10. *Statistical claims need calibration.* No threshold becomes a default without a calibration run (Phase 5).
11. *Ask the user only when blocked* (license choice, hours, access). Otherwise decide, and record the decision in DECISIONS.md.

## Known pitfalls to avoid
- Do not regress on ln|Δ_obs|: Δ is noisy, near zero, and changes sign.
- Do not gate on χ²_ν ≤ 1 + 3√(2/(N−k)): at N=3300 it allows only 7.4% excess and rejects almost all real data. Use the full likelihood. A free intrinsic scatter makes a χ² gate circular, so do not gate on it.
- Ablation pools of different size add a free BIC bonus. In v1 the bonus is 2·ln(|C_abl|/|C_dom|): +5.22, +5.18, +8.50 across the three scenarios. The blind protocol must compare candidates from the same pool.
- Near u → 0, many primitives agree through second order and differ only at the third. Expect equivalence classes in narrow windows. This is a feature, not a bug.
- Naive forms such as u/expm1(u) − 1 lose all precision as u → 0 (the same failure as the paper's example). Any new function needs a regularized small-u branch and a test at u = 1e-16.

## Deferred. DO NOT implement unless the user explicitly asks
Stage-0 information funnel, log-LASSO, 50-variable support, four new primitives (D_ulog, D_inst, D_pade12, D_algln), full numerical OED, battery and NIST datasets, GPU determinism claims, any LLM in the verdict path.

## Report format at the end of every phase
1. What was done (files and commits).
2. Commands run and their key outputs.
3. Done-when checklist with pass/fail.
4. Surprises and deviations from PREDICTIONS.
5. Decisions made (also in DECISIONS.md).
6. Open questions for the user.

## Epistemic Honesty & Provenance Protocol (Mandatory)
Every AI agent operating in this repository MUST strictly follow these rules:

1. **No False Novelty Claims:**
   - Any formula that matches or re-wraps existing literature (e.g. Churchill–Usagi 1972, CES, Milgrom, McGaugh) MUST be explicitly labeled as `origin_type="literature"` with its primary citation.
   - An agent must NEVER present an algebraic form retrieved from LLM pre-training weights as a "new discovery made by this project".
   - `is_novel_claim` may only be `True` if accompanied by an explicit, human-reviewed, first-principles derivation starting from stated physical postulates.

2. **Zero-Memory Quantitative Reporting:**
   - All numbers, BIC values, ΔBICs, and parameter values quoted in markdown reports MUST be read directly from the generated CSV/JSON artifacts.
   - Never quote parameters from memory, never change units on-the-fly without running the conversion script, and never fabricate table entries.
   - `tests/test_agent_integrity_guards.py` validates that markdown reports match CSV artifacts within strict numerical tolerances.

3. **Core Module Freeze & Architecture Isolation:**
   - The production library `src/adcd/` is locked and frozen. No experimental spikes from `spike/` may be imported into `src/adcd/`.
   - On exploratory branches (such as `Exploring`), `README.md` MUST remain untouched to prevent conflicts with mainline production branches (`PythonADCD`, `JuliaADCD`). All exploratory documentation must live in `reports/`.

4. **Oracle Leak Prevention:**
   - The auditor must evaluate candidate expressions strictly blind: inputs are only `(x, y_obs, sigma_y)`.
   - Under no circumstances may `y_true`, noiseless residuals, or branch conditionals based on `scenario_name` or `case_name` be used to alter parameter bounds, likelihood weighting, or structural family labels.
   - Programmatic guards in `tests/test_no_oracle_leak.py` enforce this automatically.

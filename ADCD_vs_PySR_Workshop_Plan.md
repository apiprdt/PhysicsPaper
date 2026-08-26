# Experiment Plan: ADCD vs PySR — Noise Robustness (Workshop)

**Pre-registration Status**: All design decisions in this document were fixed
**before** any experiment was run. Any deviation from this plan after seeing
initial results must be recorded in the Changelog (Section 8) with explicit
justification. This discipline is identical to the BIC/NMSE threshold
calibration protocol maintained throughout the ADCD audit history.

---

## 1. Thesis Under Test

**Single, sharp, workshop-scoped claim**:

> *ADCD's structured search (physics grammar + algebraically regularised
> primitives + four-gate verification protocol) recovers the correct
> functional class of the correction term at a higher rate than unconstrained
> genetic-programming search (PySR, default configuration) under increasing
> noise — and the crossover noise level at which this advantage becomes
> statistically significant can be identified empirically.*

**Not claimed**: "ADCD is more accurate in general."
**Not claimed**: "ADCD wins under all conditions."

The claim is bounded to the high-noise regime with controlled synthetic data.

---

## 2. Scope — In and Out

| Item | Status | Rationale |
|---|---|---|
| ADCD vs PySR (default), noise sweep, multi-seed | ✅ **Phase 1 (required)** | Infrastructure audited, claim sharp, produces headline figure |
| PySR + ADCD_BIC reselection (partial ablation) | ✅ **Phase 2 (if time permits)** | Low effort (post-hoc on Phase 1 Pareto fronts), partially isolates "grammar vs framework" |
| Full ADCD grammar ablation (engine rebuild) | ❌ Future work | Requires new engine construction, disproportionate to workshop scope |
| AI Feynman, PhySO | ❌ Future work | AI Feynman maintenance status uncertain; PhySO requires per-problem RL training — high timeline risk |
| PySR with `dimensional_constraint_penalty` / `TemplateExpression` | ❌ Future work | Risk of appearing as post-hoc tuning; cite as explicit future direction in Discussion |
| Real observational data (SPARC) | ❌ Not for this workshop | SPARC pipeline (M/L ratio, scatter calibration) is under active iteration — do not headline moving-target results |
| Extrapolation NMSE metric | ✅ **Phase 1 (additional diagnostic)** | Demonstrates asymptotic correctness of ADCD primitives; implemented at low cost; reported alongside recovery rate but not used for selection |

**Pre-written reviewer defence (write before asked, not after)**:

> "We compare against out-of-the-box PySR because this represents the most
> common deployment of unconstrained SR without explicit domain knowledge.
> PySR also supports `dimensional_constraint_penalty` and `TemplateExpression`
> configurations that could close part of this gap; evaluating those
> knowledge-matched configurations is explicitly designated future work."

---

## 3. Fairness Protocol

All decisions below are implemented in `eval/benchmark_pysr_comparison.py`.

| Axis | Decision | Implementation |
|---|---|---|
| **Data** | Bit-identical across methods | Single `generate_data()` call; runtime assertion verifies determinism |
| **Residual isolation** | Both methods search the correction residual (not the full law) | `detect_correction_mode()` applied to shared data; `correction_type` ground truth field is NOT read directly |
| **Feature access** | `scenario.classical_variables` only, for both methods | No extra variables given to either method |
| **PySR operators** | Default out-of-the-box set: `+, -, *, /, exp, log, sqrt, sin, cos` | Not customised per-scenario |
| **Compute budget** | PySR timeout ≥ ADCD wall-clock (floor: `MIN_PYSR_SECONDS=30`) | Measured automatically per run; bias if any is in PySR's favour |
| **PySR evaluation metric** | `classify_structure()` from `adcd.metrics`, applied to PySR's own best candidate | PySR selects its best via `model_selection="best"` (internal criterion) — not overridden by us |
| **PySR parameter count for BIC** | Counted from SymPy expression's free numeric constants, NOT from `complexity` column | `complexity` = AST node count ≠ degrees of freedom; using it would overpenalise PySR in Phase 2 |
| **Seeds** | ≥5 independent seeds per (scenario, noise) point | `DEFAULT_SEEDS = [42, 43, 44, 45, 46]` |
| **Confidence intervals** | Wilson 95% CI, not normal approximation | More accurate at 0%/100% recovery (guaranteed to occur at extreme noise) |
| **Reproducibility** | PySR runs with `deterministic=True, parallelism="serial"` | Reduces PySR throughput but enables cross-seed reproducibility; documented in paper Methods |

---

## 4. Statistical Design

- **Unit of analysis**: one run = one (scenario, noise-level, seed) triple.
  Recovery = binary (structural class matches `scenario.correction_class` or not).
- **Aggregation**: recovery rate per (scenario, noise) from ≥5 seeds, with Wilson 95% CI.
- **Primary comparison**: overlay two curves (ADCD vs PySR) with confidence bands.
  The headline claim is the **shape of the curve**, not a single win/loss number.
- **If CIs overlap at a noise point**: report "no significant difference at this
  level" — do not claim a win where CIs overlap. This is the first question any
  statistical reviewer will ask.
- **Crossover point**: the lowest noise level at which ADCD's CI is strictly
  above PySR's CI with no overlap. This is the single most important number
  for the abstract.
- **Secondary metric (extrapolation NMSE)**: evaluated on a held-out domain wider
  than training, at zero noise. Reported as a diagnostic — not used for
  selection or for defining the crossover point.

---

## 5. Execution Plan

### Phase 0 — Setup (before first run)
- [ ] Install PySR: `pip install pysr && python -c "import pysr; pysr.install()"`
- [ ] Run ONE combination manually and verify that the data arrays seen by ADCD
  and PySR are byte-identical (the runtime assertion in `build_shared_data()`
  will catch any failure automatically).
- [ ] Commit `eval/benchmark_pysr_comparison.py` and this plan document
  **before the first full run** to establish a timestamp showing the protocol
  was set before results were observed.

### Phase 1 — Main Experiment (required)
- [ ] Full run: 3 scenarios × 7 noise levels × 5 seeds = 105 combinations.
  Estimated wall-clock: ~1–3 hours (PySR 30–90 s/run).
- [ ] Sanity check: `n_equations_in_pareto` is non-zero for all PySR runs.
- [ ] Verify: determinism assertion passes for all 105 combinations (no crash).
- [ ] Generate headline 3-panel figure: recovery rate vs noise, with CI bands.
  (`--plot-only` flag regenerates figures from saved JSON without re-running.)
- [ ] Write one paragraph per scenario: at what noise level does the crossover
  occur, and what is the physical interpretation (link to SNR floor / structural
  ambiguity diagnosed in the ADCD validation report).

### Phase 2 — Secondary Experiment (if time permits)
- [ ] Rerun with `--include-ablation` (PySR Pareto fronts from Phase 1 are
  consumed; no PySR re-run needed if JSON is saved). If time-constrained,
  run ablation on a subset of noise levels around the crossover point only.
- [ ] Interpret three-row comparison: PySR default vs PySR+ADCD_BIC vs ADCD.
  If PySR+ADCD_BIC falls strictly between the other two, the ADCD framework
  (not only its grammar) contributes independently.

### Phase 3 — Writing
- [ ] Methods section: write directly from Section 3 of this document.
  Do not paraphrase from memory.
- [ ] Limitations section: include all "❌ Future work" items from Section 2.
- [ ] Pre-submit review: have one person not involved in coding read Section 3
  and explain the protocol back in their own words.

---

## 6. Anticipated Reviewer Questions

| Question | Pre-prepared answer |
|---|---|
| "Why not compare against PySR with unit constraints?" | See Section 2 pre-written defence — explicit future work, not avoided |
| "Was PySR given enough compute?" | Budget ≥ ADCD wall-clock guaranteed per run (Section 3) |
| "Why search the residual instead of the full law?" | Controls the confound of "easier problem" — both methods solve exactly the same fitting task (Section 1 & 3) |
| "Does this generalise beyond these 3 scenarios?" | No — explicitly acknowledged as a limitation; scope is intentionally narrow for a workshop |
| "What if PySR wins at low noise?" | Report as-is — it strengthens the story ("ADCD advantage is noise-regime-specific, not a blanket claim") |
| "Are the recovery rates stable across seeds?" | Yes — reported with Wilson 95% CI from ≥5 seeds, not a single-seed result |
| "Why serial PySR?" | Required for deterministic reproducibility; timeout guarantee prevents time-starvation (Section 3) |

---

## 7. Definition of Done

- [ ] `run_outputs/pysr_comparison.json` contains complete results for all
  105 Phase 1 combinations, committed to the repository.
- [ ] `run_outputs/fig_comparison_recovery.{pdf,png}` exists and is ready for
  paper inclusion.
- [ ] `run_outputs/fig_comparison_extrap.{pdf,png}` exists (extrapolation figure).
- [ ] Crossover noise level table (one row per scenario) is written.
- [ ] Limitations draft includes all scope-out items from Section 2.
- [ ] (Phase 2, optional) `ablation_recovery` column present in summary JSON
  with one interpretive paragraph.

---

## 8. Changelog

*Record every deviation from this plan that was made after seeing any
experimental results. An empty table here means no deviations occurred.*

| Date | Change | Justification |
|---|---|---|
| — | — | — |

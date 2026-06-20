# Changelog

All notable changes to **ADCD** are documented below. ADCD adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.1] — 2026-06-20

### Added
- **SPARC MOND validation**: rigorous galaxy-level cross-validation (10 repeated 50/50 train/test splits), 50-resample bootstrap parameter CIs, and three-tier quality-cut robustness study on the SPARC sample
- **Bootstrap CI table** (`tab_sparc_bootstrap.tex`) with display symbols $\theta_0$, $\theta_1$ rendered in canonical order
- **Paper text**: parameter-degeneracy discussion (deep-MOND $\hat\theta_0\sqrt{\hat\theta_1}$ invariance) added to bootstrap and robustness sections

### Changed
- **Lelli et al. SPARC citation corrected**: *The Astrophysical Journal* 836:152 (2017) → *The Astronomical Journal* 152:157 (2016)
- **Lint/format config aligned**: `[tool.black]` line-length raised 100 → 120 to match `.flake8`; removed redundant `[tool.flake8]` block from `pyproject.toml`
- **Version strings synced to 2.2.1** across `README.md`, `CITATION.cff`, `.zenodo.json`, `docs/index.md`, `docs/paper.md`
- **Test counts updated** 95/77 → **116** in README badge, project structure, docs hero stats, and installation guide
- **Docs hero stat** corrected: stale "+44.5 pp over PySR" → **+77.8 pp**

### Fixed
- **CI lint gate**: 7 flake8 errors in `src/adcd/experiments/sparc_robustness.py` resolved; CI lint step now passes clean
- **`scripts/generate_sparc_tables.py`**: bootstrap row symbols now map `theta_r1_0`/`theta_r1_1` → $\theta_0$/$\theta_1$

---

## [2.2.0] — 2026-06-18

### Added
- **Phase 2 — Multivariable Correction Discovery**: Four new core modules:
  - `buckingham_pi.py` — `BuckinghamPiEngine`: nullspace-based Buckingham Π group generator from dimensional matrix
  - `sequential_arc.py` — `SequentialARCChecker`: per-variable independent ARC limit checking
  - `residual_factorizer_v2.py` — `ResidualFactorizerV2`: variance-decomposition separability detection (multiplicative / additive / none)
  - `multivar_orchestrator.py` — `MultivariableOrchestrator`: end-to-end multivariable correction search pipeline
- **Phase 2 benchmark**: 2/4 multivariable scenarios solved (50% from baseline 0/4) on Yukawa Mass-Ratio and Turbulent Drag
- **New test files**: `test_multivar_arc.py`, `test_phase2_components.py`, `test_bayesian_ranker.py`, `test_identifiability.py`, `test_gate_telemetry.py`
- **LaTeX table overflow fixes**: All `\hbox` overfull warnings resolved in `tab_pysr_config.tex`, `tab_runtime.tex`, `tab_template_leakage.tex`, and `main.tex` (paper compiles cleanly at 725 KB)

### Fixed
- **Flake8 CI**: Removed all 40+ unused imports and unused variable warnings (F401, F841, E111, E272) across `src/` and `tests/`

---

## [2.1.3] — 2026-06-11

### Changed
- **Paper Polish (15 Audit Fixes)**: Corrected single-gate claim in Introduction; updated Figure 4 caption and Table 11 notes to match data exactly; disclosed `correction_mode="auto"` in Methodology; cleaned up Muon g-2 literal constant to $\alpha/\pi$ in Table 8 recovery parameters; clarified PySR hall-of-fame definition and Table 5 footnote wrapping.
- **LaTeX Compilation**: Resolved all undefined references and labels.

---

## [2.1.2] — 2026-06-10

### Added
- **`docs/SUBMISSION_CHECKLIST_v2.1.2.md`**: step-by-step GitHub Release, Zenodo, and arXiv submission guide
- **`scripts/verify_paper_claims.py`**: PySR fair 77.8 pp gap guard at 5% noise

### Changed
- **Paper narrative polish (8/10 path)**: evaluation regimes paragraph (Primary / Supplementary / Real-world); quantitative claims reframed (structural lead, Blackbody NMSE qualifier); ARC collective-filter framing aligned with ablation
- **Multi-seed benchmark refreshed (Tier B+)**: mean structural recovery **82.8% ± 7.7%** (was 81.1% ± 10.3% on pre-fix pipeline)
- **Abstract PySR comparison** corrected to **77.8 percentage-point gap** vs PySR fair at 5% noise (was incorrectly 44.4 pp from legacy fast profile)
- **README** PySR table updated to fair profile; Mercury NMSE corrected to $1.11 \times 10^{-5}$

### Fixed
- **LaTeX table generators**: `tab_pysr_config.tex` row endings (`\\\\`); `tab_pulsar_sensitivity.tex` underscore escaping and math mode
- **`gate_telemetry.json`** refreshed via `run_correction_discovery.py --proposer mock`

---

## [2.1.1] — 2026-06-10

### Added
- **Related Work**: PhySO (Tenachi et al. 2023) and LaSR (Grayeli et al. 2024) positioning paragraphs
- **`hybrid_seed42_results.json`**: frozen Hybrid Proposer benchmark (33/36 = 91.7% at seed=42)
- **`docs/ZENODO_RELEASE_v2.1.0.md`**, **`docs/GITHUB_RELEASE_v2.1.0.md`**

### Changed
- **Paper tone**: `prune` → `filter`, `guarantee` → `screen` with dimensional-relaxation qualifier (\Cref{sec:limitations})
- **Package version** synced to 2.1.0 in `pyproject.toml` and `__init__.py`
- **`reproduce_all.ps1`**: step 8 replaced with `pytest tests/test_real_data.py -k mercury`

### Fixed
- **Paper statistics** aligned with frozen `reproducibility_results.json` (seed disclosure, per-seed rates)
- **Binary pulsar** framing: sensitivity study separated from main 4/4 real-world headline

---

## [2.1.0] — 2026-06-09

### Added
- **Binary pulsar v2.1** reduced-variable benchmark (fixed M, a, e; varying P) with Peters prefactor helper
- **`run_binary_pulsar_sensitivity.py`**: P_only / P_e / P_e_M / full variant study for reviewer-facing ablation
- **`scripts/generate_real_world_tables.py`**: auto-generates parameter recovery, template leakage, and sensitivity LaTeX tables
- **`tests/test_metrics_scale.py`**: guards against scale-adaptive NMSE regressions on sub-nano residuals

### Changed
- **Real-world reporting** now separates structural (5/5), quantitative NMSE $< 10^{-4}$ (3/5), and optimizer convergence (2/5)
- **`evaluate_correction`**: SymPy `lambdify` evaluation replaces fragile `eval()` string substitution
- **Stage-2 BIC reranking** uses post-hoc validated NMSE from `evaluate_correction`, not optimizer-internal scores

### Fixed
- **Scale-adaptive NMSE** in `jax_optimizer.py` and `metrics.py` (fixed $\varepsilon=10^{-10}$ floor caused false convergence on binary pulsar data $\sim 10^{-15}$)
- **Binary pulsar false positive**: degenerate $\theta_0\to 0$ models no longer win BIC with NMSE $= 1.0$ while `class_match=true`

---

## [2.0.0] — 2026-06-08

### Added
- **Parameter-scaled L-BFGS-B optimizer** (`jax_optimizer.py`): each restart now normalises variables to O(1) by dividing by their initial values, completely eliminating gradient underflow on extreme-scale parameters (e.g. `G = 6.67e-11`)
- **Mixed log-uniform initialisation**: 50/50 blend of narrow `[-6, 6]` and wide `[-20, 20]` exponent ranges for better coverage of both standard and astrophysical/quantum scales
- **`loss_mode='auto'`** with dynamic-range threshold `1e4`: high-DR scenarios (e.g. Blackbody, DR ≈ 7.1e4) automatically switch to full-reconstruction loss; all 9 standard scenarios remain on residual loss
- **Negative power-law templates** added to `CorrectionMockProposer`: `-(θ₀/v₁)⁴`, `-(θ₀/v₁)^θ₁`, `θ₀(θ₁/v₁)⁴ − 1`, `-(θ₀/v₁)^θ₁ + θ₂` — fixes Net Radiation 4/4 discovery
- **Degenerate exponent detection** in `classify_structure`: power-law expressions where `θ ≈ 1.0` are reclassified as polynomial, avoiding false class mismatches on Muon g-2
- **AST node-count tie-breaker** in BIC sort (`correction_orchestrator.py`): prefers structurally simpler expressions when BIC scores are equal
- **`n_restarts = 15`** (up from 5) in JAXOptimizer for improved global search coverage
- **58 automated unit tests** covering all pipeline gates, optimizer, proposer, and public API
- **`run_real_data_benchmark.py`** extended with Mercury Perihelion, Lamb Shift, Muon g-2, and Blackbody scenarios
- **`run_reproducibility.py`** multi-seed study: 5 seeds × 9 scenarios × 4 noise levels = 180 runs

### Changed
- `DYNAMIC_RANGE_THRESHOLD` raised from `1e3` → `1e4` (prevents Screened Coulomb 5%/10% regression)
- `MockProposer` now injects `theta_0 * {v1} / {const}` templates in physical-constant injection phase
- Benchmark table in `README.md` corrected to use actual scenario names
- `.gitignore` updated: ignores `scratch/`, `scratch_*`, `baseline_pre_fix.json`, `*.zip`, `*.tar.gz`

### Fixed
- **Net Radiation (0/4 → 4/4)**: negative power-law correction `-(T_env/T)⁴` now discovered reliably at all noise levels
- **Screened Coulomb 5%/10% regression**: dynamic-range threshold fix restores `residual` loss mode for standard scenarios
- **Blackbody structural match**: `loss_mode='auto'` correctly switches to full-reconstruction loss for DR ≈ 7.1e4

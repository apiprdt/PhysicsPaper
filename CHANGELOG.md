# Changelog

All notable changes to ADCD will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.3] — 2026-06-11

### Changed
- **Paper Polish (15 Audit Fixes)**: Corrected single-gate claim in Introduction; updated Figure 4 caption and Table 11 notes to match data exactly; disclosed `correction_mode="auto"` in Methodology; cleaned up Muon g-2 literal constant to $\alpha/\pi$ in Table 8 recovery parameters; clarified PySR hall-of-fame definition and Table 5 footnote wrapping.
- **LaTeX Compilation**: Resolved all undefined references and labels.

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
- Benchmark table in `README.md` corrected to use actual scenario names (previously listed wrong scenario names)
- `.gitignore` updated: ignores `scratch/`, `scratch_*`, `baseline_pre_fix.json`, `*.zip`, `*.tar.gz`

### Fixed
- **Net Radiation (0/4 → 4/4)**: negative power-law correction `-(T_env/T)⁴` now discovered reliably at all noise levels
- **Screened Coulomb 5%/10% regression**: dynamic-range threshold fix restores `residual` loss mode for standard scenarios
- **Blackbody structural match**: `loss_mode='auto'` correctly switches to full-reconstruction loss for DR ≈ 7.1e4

---

## [1.1.0] — 2026-06-04

### Added
- `src/adcd/` installable package structure with `__init__.py` public API
- `pyproject.toml` for PEP 517/518 build system (`pip install adcd`)
- GitHub Actions CI workflow: test suite on Python 3.10 + 3.11, LaTeX paper compilation
- GitHub Actions publish workflow: Trusted Publishing to PyPI on release
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR templates
- `LICENSE` (MIT)
- `CHANGELOG.md` (this file)

### Changed
- All internal imports migrated from `from src.X` to `from adcd.X`
- `README.md` updated with installation badge, Quick Start, and BibTeX citation

### Fixed
- Overclaims corrected in paper: "guarantees" → "strongly enforces"; ARC/gate individual ablation clarified; timing claim removed
- Author email `maeapip10@gmail.com` added to paper and Zenodo metadata

---

## [1.0.0] — 2026-06-04 (Initial Zenodo Release)

### Added
- Full ADCD discovery pipeline: AST complexity gate → dimensional checker → transcendental guardrail → ARC asymptotic gate → JAX L-BFGS-B optimizer → BIC reranker
- 9-scenario benchmark suite: Relativistic KE, Yukawa Gravity, Anharmonic Spring, Screened Coulomb, Net Radiation, Nonlinear Drag, Mystery-A, Mystery-B, Mystery-C
- PySR baseline comparison (22.2–66.7% vs ADCD 88.9–100%)
- MLP baseline comparison (NMSE: 8.56e-5 at 0% noise vs ADCD 5.51e-12)
- Blind generalization test: Van der Waals, Stokes-Einstein, Wien displacement
- Ablation study: gates, BIC reranking
- LaTeX paper with full reproducibility data
- Zenodo DOI: `10.5281/zenodo.20534940`

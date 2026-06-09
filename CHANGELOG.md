# Changelog

All notable changes to ADCD will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

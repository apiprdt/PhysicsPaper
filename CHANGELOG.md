# Changelog

All notable changes to the ADCD project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-26
### Added
- Core 8-primitive asymptotic dictionary engine (`D_lor`, `D_rat`, `D_exp`, `D_log`, `D_sqrt_inv`, `D_pow`, `D_osc`, `D_sat`).
- 5-gate screening cascade (Dimensional, Asymptotic Regime, Mode, Curvature, QuickFit).
- Stage 2 JAX L-BFGS-B optimizer in float64 with domain-aware boundary constraints.
- Extended BIC model selection scoring including search-space complexity penalty.
- Canonical physical scenarios: Screened Coulomb, Entropy Expansion, Time Dilation.
- Automated 4-step blind validation protocol (`budget_disclosure`, `positive_control`, `ablation_control`, `determinism_check`).
- SPARC galactic rotation curve real astrophysics data loaders.
- Interactive end-to-end quickstart tutorial notebook.

## [1.1.0-dev] - 2026-11-08 (Branch ExploreSmth)
### Changed
- Fixed undefined `Union` type annotation import in `quickfit.py`.
- Fixed unused imports and string formatting issues identified by `flake8`.
- Added comprehensive regression tests `tests/test_regression_v1.py` covering single-seed and 5-seed reference tables.
- Added `REPO_MAP.md`, `DECISIONS.md`, and `AGENTS.md` project rules for Phase 0 baseline pinning.

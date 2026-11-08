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
### Added
- Phase 1 `ClaimCard` schema (`src/adcd/claim_card.py`) with Pydantic v2 validation, noise specifications, and canonical SHA-256 hashing.
- Phase 1 Audit API (`src/adcd/audit.py`): `audit()`, `audit_custom_expressions()`, and dual symbolic/numerical `verify_classical_limit()`.
- Phase 1 deterministic `AuditReport` with plain-language verdict semantics.
- Phase 2.1 Joint baseline-correction optimization and Hessian condition number analysis (`src/adcd/joint_fit.py`).
- Phase 2.2 Scale-invariant Stage 1 candidate screening with discrete log sweep and closed-form WLS amplitude solution (`src/adcd/scale_invariant_screening.py`).
- Phase 2.3 Heteroscedastic Gaussian likelihood and intrinsic scatter estimation (`src/adcd/likelihood.py`).
- Unit tests for all Phase 2 modules (`tests/test_joint_fit.py`, `tests/test_scale_invariant_screening.py`, `tests/test_likelihood.py`).
- Smoke testing suite for quantum known-answer cases (`scripts/smoke_quantum.py`, `reports/smoke_quantum.md`).
- Comprehensive regression tests `tests/test_regression_v1.py` covering single-seed and 5-seed reference tables.
- Phase 3 Blind Three-Tier Protocol & Equal-Pool Self-Ablation (`src/adcd/validation_v2.py`) issuing `UNIQUELY IDENTIFIABLE`, `EQUIVALENCE CLASS`, or `WITHHELD` with zero ground-truth references.
- Equal-pool self-ablation cancelling the $+5.18$ search-space penalty bias, shifting Entropy Expansion to honest `EQUIVALENCE CLASS`.
- Shared low-order Taylor series analysis explaining mathematical indistinguishability within equivalence classes.
- Flexible cubic spline check for residual model misspecification.
- Unit tests for Three-Tier Protocol and zero-ground-truth verification (`tests/test_validation_v2.py`).
- Phase 3 rerun evaluation script (`scripts/run_phase3_rerun.py`), results (`reports/phase3_rerun_results.json`), and verdict change table (`reports/verdict_change_table.md`).
- Unit tests for ClaimCard, canonical hashing, and limit verification (`tests/test_claim_card.py`).
- Added `REPO_MAP.md`, `DECISIONS.md`, and `AGENTS.md` project rules for Phase 0 baseline pinning.
### Changed
- Fixed undefined `Union` type annotation import in `quickfit.py`.
- Fixed unused imports and string formatting issues identified by `flake8`.

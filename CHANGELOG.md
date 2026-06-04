# Changelog

All notable changes to ADCD will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- 9-scenario benchmark suite covering relativistic KE, Yukawa gravity, anharmonic spring, Debye heat capacity, quantum sinc diffraction, Stokes drag, Maxwell-Boltzmann, quantum tunneling, screened Coulomb
- PySR baseline comparison (22.2–66.7% vs ADCD 88.9–100%)
- MLP baseline comparison (NMSE: 8.56e-5 at 0% noise vs ADCD 5.51e-12)
- Blind generalization test: Van der Waals, Stokes-Einstein, Wien displacement
- Ablation study: gates, BIC reranking
- LaTeX paper with full reproducibility data
- Zenodo DOI: `10.5281/zenodo.20534940`

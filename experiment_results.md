# Experiment Report: ADCD Correction Discovery

> **AUTO-GENERATED** — do not edit manually.
> Generated: 2026-06-09 17:04 UTC | git: `7724dbe`

Regenerate: `python scripts/generate_experiment_report.py`

## Standard Benchmark (9 scenarios × 4 noise levels)

- **Class match (9-scenario subset):** 34/36 (94.4%)
- **0% noise:** 9/9 (100.0%)
- **1% noise:** 9/9 (100.0%)
- **5% noise:** 8/9 (88.9%)
- **10% noise:** 8/9 (88.9%)

### Search Space Telemetry (ADCD)

| Metric | Count |
|--------|------:|
| Candidates proposed | 3,225 |
| Gate pipeline inputs | 3,225 |
| Parse failures | 0 |
| AST rejections | 0 |
| Dimensional rejections | 337 |
| Transcendental rejections | 0 |
| ARC rejections | 2,233 |
| Coarse NMSE rejections | 0 |
| Stage 1 survivors (unique per iter) | 607 |
| Stage 2 optimization calls | 598 |
| Gate output (passed all) | 655 |
- **Overall gate survival:** 20.3%

## Real-World Validation (synthetic-real hybrid)

| Scenario | Class Match | Converged | Full NMSE |
|:---|:---:|:---:|:---:|
| Real: Mercury Perihelion | ✓ | — | 1.11e-05 |
| Real: Hydrogen Lamb Shift | ✓ | ✓ | 1.69e-18 |
| Real: Blackbody Radiation | ✓ | — | 2.59e-02 |
| Real: Muon g-2 | ✓ | ✓ | 7.94e-07 |
| Real: Binary Pulsar Decay | ✓ | — | 3.94e-03 |

**Summary:** 5/5 structural class matches; 3/5 quantitative (NMSE $< 10^{-4}$); 2/5 optimizer converged ($< 10^{-5}$).

> Data are synthetic-real hybrid (JPL/NIST/CODATA constants), not raw instrument archives.
> Template-assisted (mock) vs zero-shot (hybrid/gemini) results must be reported separately.
> Binary pulsar v2.1 uses reduced-variable formulation (P only); see sensitivity study.

## PySR Baseline Comparison

- **PySR fast (legacy):** 16/36 class matches (44.4%), mean 2.9s/scenario, mean hall-of-fame size 0
- **PySR fair:** 15/36 class matches (41.7%), mean 8.8s/scenario, mean hall-of-fame size 14

## Reproducibility

- Entries in `reproducibility_results.json`: 180

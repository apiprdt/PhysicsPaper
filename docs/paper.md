# The ADCD Paper

ADCD was introduced in the following preprint paper:

> **Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery**  
> *Muhammad Afif Erdita (2026)*  
> DOI: [10.5281/zenodo.20534940](https://doi.org/10.5281/zenodo.20534940)  
> arXiv: [Under Submission]

---

## Abstract

We present **Anomaly-Driven Correction Discovery (ADCD)**, a novel symbolic regression framework tailored for evolutionary scientific discovery. Unlike classical symbolic regression which searches the entire mathematical space from scratch (*tabula rasa*), ADCD mimics the historical evolution of scientific theories by learning *dimensionless correction terms* ($\Delta$) relative to established classical baselines. By integrating domain-specific constraints directly into a cascaded filtering pipeline — consisting of Abstract Syntax Tree (AST) complexity limits, dimensional homogeneity verification, transcendental argument guardrails, and asymptotic boundary consistency (ARC) gates — ADCD eliminates physically unplausible equations before numerical optimization. Free parameters are optimized using a JAX-traced, multi-restart, parameter-scaled L-BFGS-B optimizer, and selected using the Bayesian Information Criterion (BIC). We demonstrate that ADCD achieves **82.8% (±7.7%) mean structural recovery** across 5 independent seeds, significantly outperforming tabula rasa baselines like PySR under noisy regimes (+77.8 pp at 5% noise). Furthermore, ADCD successfully recovers correct structural classes on historical physical anomalies, including the general relativistic correction to Mercury’s perihelion, Schwinger's QED correction to the muon $g-2$, the Lamb Shift, and Planck's blackbody radiation law.

---

## Key Methodology Components

```
                +----------------------------+
                |    Observed Anomaly Data   |
                +--------------+-------------+
                               |
                               v
                +----------------------------+
                |  Residual Extraction (Δ)   |
                +--------------+-------------+
                               |
                               v
                +----------------------------+
                | Template Proposer Cascade  |
                +--------------+-------------+
                               |
                               v
                +----------------------------+
                |   Cascaded Physics Gates   |
                | - AST Complexity Limit     |
                | - Dimensional Homogeneity  |
                | - Transcendental Guardrail |
                | - Asymptotic Limits (ARC)  |
                +--------------+-------------+
                               |
                               v
                +----------------------------+
                | Parameter-Scaled JAX Fit   |
                +--------------+-------------+
                               |
                               v
                +----------------------------+
                |      BIC Model Ranking     |
                +----------------------------+
```

## How to Cite

If you use ADCD or refer to the paper benchmarks in your academic work, please use the following BibTeX entry:

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained
                Symbolic Regression for Evolutionary Scientific Discovery}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.1.3},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```

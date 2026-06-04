# Anomaly-Driven Correction Discovery (ADCD)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

ADCD is a physics-informed symbolic regression framework designed to discover mathematical corrections to classical laws from anomalous experimental data. Rather than learning entire equations from scratch, ADCD integrates known physical constraints (dimensional homogeneity, asymptotic limits, and transcendental arguments) to guide the search for parsimonious, mathematically sound, and physically plausible correction terms ($\Delta$).

For more detail, please read our preprint: **[Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery](paper/main.pdf)**.

---

## Repository Structure

```
├── paper/
│   ├── main.tex                 # LaTeX Source Document
│   ├── main.pdf                 # Compiled Scientific Paper
│   └── figures/                 # Evaluation plots (PDF/PNG format)
├── src/                         # Python Core Library
│   ├── anomaly_scenarios.py     # Definitions of synthetic physical anomalies
│   ├── real_data_loader.py      # Real experimental loaders (DE440, NIST, Fermilab)
│   ├── real_scenarios.py        # AnomalyScenario wrappers for real data
│   └── ...                      # Core proposer and verification filters
├── tests/                       # Unit Test Suite
│   ├── test_real_data.py        # Tests for real physics experiments
│   └── ...                      # Model checking tests
├── requirements.txt             # Python Package Dependencies
├── run_experiments.py           # Main benchmark runner on synthetic scenarios
├── run_real_data_benchmark.py   # Main benchmark runner on real-world anomalies
└── run_timing_analysis.py       # Computational timing analysis tool
```

---

## Installation & Setup

1. **Python version**: Requires Python 3.11+.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Guide

### 1. Run Unit Tests
To verify all loaders, scenarios, and filters work correctly:
```bash
pytest tests/ -v
```

### 2. Run Synthetic Benchmarks
To reproduce the results for the 9 synthetic anomaly scenarios (Textbook, Cross-Domain, and Novel tiers):
```bash
python run_experiments.py
```

### 3. Run Real-World Physics Benchmarks
To run the correction discovery on real physical anomalies (Mercury Precession, Hydrogen Lamb Shift, Blackbody Radiation, and Muon g-2):
```bash
python run_real_data_benchmark.py
```

### 4. Run Computational Timing Analysis
To run the timing analysis comparing ADCD components to PySR:
```bash
python run_timing_analysis.py
```

---

## License

The code is licensed under the MIT License. The paper and figures are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.

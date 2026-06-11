# PySR Comparison

To benchmark ADCD against state-of-the-art symbolic regression, we compared it directly with **PySR** (using the default PyJuliana backend).

## Experimental Setup

- **PySR Config**: Evaluated using the `fair` profile: 100 search iterations, maxsize of 30, and a 60-second runtime limit per run (mirroring a reasonable search budget).
- **Target**: PySR was provided with the exact same residual $y_{\text{obs}} - y_{\text{classical}}$ as ADCD, ensuring a fair baseline comparison.
- **Evaluation**: Counts a success only if PySR recovers the correct structural equation class.

## Performance Comparison (seed=42)

| Method | 0% Noise | 1% Noise | 5% Noise | 10% Noise |
|---|:---:|:---:|:---:|:---:|
| **ADCD (ours)** | **9/9 (100%)** | **9/9 (100%)** | **8/9 (88.9%)** | **8/9 (88.9%)** |
| PySR fair | 4/9 (44.4%) | 5/9 (55.6%) | 1/9 (11.1%) | 5/9 (55.6%) |

ADCD outperforms PySR by **+77.8 percentage points** at 5% noise (88.9% vs 11.1%).

## Key Takeaways

1. **A Priori Constraints**: By filtering candidate templates before numerical fitting, ADCD avoids optimizing parameters for unphysical expressions, preserving search bandwidth.
2. **Robustness to Overfitting**: PySR often finds overparameterized mathematical expressions that fit the noisy data better numerically but are physically incorrect. ADCD’s physics gates and BIC reranking prevent this.
3. **Speed**: ADCD's cascaded checks reduce the number of optimization loops, making the overall pipeline significantly faster than tabula rasa search.

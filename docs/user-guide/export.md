# Export & Visualization

Once a correction is discovered, ADCD provides built-in utilities to export formulas, parameters, and plots.

---

## 1. LaTeX Export
To present discovered equations in publications, you can retrieve the LaTeX representation of the best correction or the entire combined law:

```python
# Returns the LaTeX of the correction term: e.g. \theta_0 \cdot \left(\frac{v}{c}\right)^2
print(result.export_latex(full_equation=False))

# Returns the combined equation: e.g. E = \frac{p^2}{2m} + \theta_0 \cdot \left(\frac{v}{c}\right)^2
print(result.export_latex(full_equation=True))
```

---

## 2. Interactive Matplotlib Plots
The `plot_residuals` method generates a publication-ready plot showing:
1. **Top Panel**: Observed data vs. the Classical baseline and the ADCD Fit.
2. **Bottom Panel**: The residual discrepancy and the fitted correction curve $\Delta$.

```python
result.plot_residuals(save_path="docs/my_fit.png")
```

---

## 3. Raw Data Summary
You can print an aligned table of all evaluated candidates that survived the gates, ranked by BIC:

```python
result.summary()
```

Output:
```text
============================================================
           ADCD CORRECTION DISCOVERY SUMMARY
============================================================
Best expression: theta_0 * (v/c)**2
True class:      polynomial (✓ Match)
Parameters:      theta_0 = 0.50021

Ranked Candidates:
1.  theta_0 * (v/c)**2           BIC=-321.4  NMSE=1.11e-05 (Best)
2.  theta_0 * (v/c)**3           BIC=-152.1  NMSE=4.22e-03
3.  theta_0 * exp(-c/v)          BIC= -88.9  NMSE=1.20e-02
============================================================
```

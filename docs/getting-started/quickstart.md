# Quickstart

If you have custom experimental data and a known classical law, you can find the correction in seconds using `adcd.fit`.

Here is a simple example showing how to discover a hidden quadratic correction ($\Delta = 0.5 x^2$) to a linear classical baseline ($y_{\text{classical}} = 2x$):

```python
import numpy as np
import adcd

# 1. Generate synthetic data
x = np.linspace(1.0, 5.0, 100)
X = {"x": x}

y_classical = 2.0 * x
y_observed  = 2.0 * x + 0.5 * x**2   # True correction is 0.5 * x^2

# 2. Fit the anomaly using ADCD
result = adcd.fit(
    X=X,
    y_obs=y_observed,
    y_classical=y_classical,
    limit_variable="x",
    limit_direction="0",
    correction_mode="additive"
)

# 3. View the results
result.summary()

# 4. Access individual components
print(f"Best Expression: {result.best_expr}")
print(f"LaTeX Representation: {result.export_latex()}")
print(f"Parameters: {result.best_theta}")
print(f"BIC Score: {result.best_bic:.2f}")

# 5. Plot the fit and residuals
result.plot_residuals()
```

## What's Happening Under the Hood?

1. **Residual Extraction**: ADCD extracts the residual $y_{\text{obs}} - y_{\text{classical}}$.
2. **Template Proposing**: The proposer (e.g. `mock` or `hybrid`) proposes candidate correction templates.
3. **Physics Gates**: The candidate templates are passed through the gate cascade:
   - **AST Complexity Gate**: Ensures formulas aren't overcomplicated.
   - **Dimensional Homogeneity Gate**: Checks if units are mathematically consistent.
   - **Asymptotic Limit (ARC) Gate**: Checks if the correction behaves correctly under physical boundary conditions (e.g., vanishes when $x \to 0$).
4. **JAX Optimization**: Parameters $\boldsymbol{\theta}$ of surviving templates are optimized using JAX-accelerated L-BFGS-B.
5. **Model Selection**: Surviving candidates are ranked using the Bayesian Information Criterion (BIC).

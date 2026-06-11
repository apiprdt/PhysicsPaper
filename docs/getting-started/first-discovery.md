# Your First Discovery

ADCD comes pre-packaged with several standard physics benchmark scenarios. Let's run a full discovery on the **Relativistic Kinetic Energy** benchmark.

In this scenario:
- The classical baseline is the Newtonian kinetic energy: $E_{\text{classical}} = \frac{p^2}{2m}$
- The observed data comes from relativistic mechanics: $E_{\text{observed}} = \sqrt{p^2 c^2 + m^2 c^4} - m c^2$
- The goal is to discover the correction term $\Delta$.

Save the following script as `first_discovery.py` and run it:

```python
import adcd

# 1. Load the Relativistic Kinetic Energy benchmark scenario
scenarios = adcd.get_all_scenarios()
relativistic_ke = scenarios[0]

print(f"Scenario Name: {relativistic_ke.name}")
print(f"Classical Equation: {relativistic_ke.expr_classical_str}")

# 2. Run the discovery orchestrator using the template proposer
result = adcd.discover_correction(
    scenario=relativistic_ke,
    max_iterations=5,
    proposer="mock"
)

# 3. Print the results
print("\n--- Discovery Complete ---")
print(f"Discovered Correction expression: {result.best_expr}")
print(f"Optimized Parameters: {result.best_theta}")
print(f"Residual NMSE: {result.best_nmse_residual:.2e}")
print(f"LaTeX expression: {result.export_latex()}")

# 4. Show the plot
result.plot_residuals()
```

When you run this script, ADCD will output status logs for each iteration. It will show candidates being filtered by the physics gates, optimized by JAX, and selected based on BIC. At the end, a Matplotlib window will pop up showing the fit and residuals!

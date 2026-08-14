import sys; sys.path.insert(0, 'src')
import numpy as np
from adcd.anomaly_scenarios import get_all_scenarios

scenarios = {s.name: s for s in get_all_scenarios()}
sc = scenarios['Entropy Expansion']

# Case 1: get_scenario_data (domain_max=1.0, noise_level=0.00)
X, y_obs, y_cls, residual = sc.generate_data(seed=42, noise_level=0.00, domain_max=1.0)
print("=== Plotting: get_scenario_data (domain_max=1.0, clean) ===")
print(f"  residual (delta_true) : {residual.min():.4f} to {residual.max():.4f}")

# Case 2: evaluate theta_fit prediction on same X
theta_0 = 8.302797192710901
dV = X['dV']
Vi = X['V_i']
delta_pred = theta_0 * np.log(1.0 + dV / Vi)
print(f"  delta_pred (theta=8.3): {delta_pred.min():.4f} to {delta_pred.max():.4f}")
print(f"  Scale ratio           : {delta_pred.max()/max(abs(residual.max()), 1e-9):.1f}x")

# Case 3: what did the validation pipeline use?
X3, _, _, residual3 = sc.generate_data(seed=42, noise_level=0.01, domain_max=1.0)
print()
print("=== Validation pipeline (domain_max=1.0, noise_level=0.01) ===")
print(f"  residual range: {residual3.min():.4f} to {residual3.max():.4f}")

# Case 4: no domain_max (what if validation didn't pass domain_max?)
X4, _, _, residual4 = sc.generate_data(seed=42, noise_level=0.01)
print()
print("=== Validation pipeline (no domain_max) ===")
print(f"  residual range: {residual4.min():.4f} to {residual4.max():.4f}")

print()
print("=== Classical constants ===")
print(sc.classical_constants)
print("=== Correction expr ===")
print(sc.correction_expr)

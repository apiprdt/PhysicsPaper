# Core Concepts

ADCD is built on two primary pillars: the **Correction-First Paradigm** and **Physics-Constrained Optimization**.

---

## 1. The Correction-First Paradigm

In classical symbolic regression, the search algorithm searches the entire space of mathematical expressions to find a function $f(\mathbf{x})$ that maps inputs $\mathbf{x}$ to targets $y$:

$$
y \approx f(\mathbf{x})
$$

This is a *tabula rasa* search, and its computational complexity grows exponentially with the size (number of operators/variables) of $f(\mathbf{x})$.

ADCD assumes that we already possess a **classical baseline theory** $y_{\text{classical}} = g(\mathbf{x})$ that is structurally correct in some limit (e.g., low velocity, low temperature, or weak coupling), but deviates from observations in extreme regimes. Instead of searching from scratch, ADCD searches for the dimensionless correction $\Delta$:

### Additive Mode
$$
y_{\text{obs}} = y_{\text{classical}} + \theta_{\text{scale}} \cdot \Delta\left(\frac{\mathbf{x}}{\mathbf{x}_0};\,\boldsymbol{\theta}\right)
$$

### Multiplicative Mode
$$
y_{\text{obs}} = y_{\text{classical}} \cdot \left(1 + \theta_{\text{scale}} \cdot \Delta\left(\frac{\mathbf{x}}{\mathbf{x}_0};\,\boldsymbol{\theta}\right)\right)
$$

By looking only for the residual $\Delta$, ADCD reduces the required expression complexity of the search target from high (e.g. 15–20 tokens for a full relativistic law) to low (e.g. 3–5 tokens for the correction factor like $(v/c)^2$).

---

## 2. Parameter-Scaled JAX Optimizer

Symbolic templates contain free parameters $\boldsymbol{\theta}$ that must be fit to the data. Standard optimizers can fail if parameters have widely different scales. 

ADCD implements a custom **JAX-traced L-BFGS-B optimizer** with:
- **Parameter scaling**: Rescales the gradient steps based on parameter magnitudes to avoid numerical stiffness.
- **Multi-restart initialization**: Spawns multiple optimization runs using log-uniform parameter initializations to find the global minimum and avoid local traps.
- **Autodiff gradients**: Automatically traces SymPy-derived templates into highly optimized JAX code, allowing GPU/TPU acceleration and exact analytical gradients.

# Configuration Options

ADCD offers flexible configuration parameters that let you balance search depth, computational budget, and verification rigor.

## Core Fitting Options

These parameters can be passed directly to `adcd.fit` or `adcd.discover_correction`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `correction_mode` | `str` | `"auto"` | `"additive"`, `"multiplicative"`, or `"auto"`. Under `"auto"`, ADCD evaluates the variance and scales of both modes to select the optimal one. |
| `max_iterations` | `int` | `5` | The number of proposal-evaluation loops. |
| `max_complexity` | `int` | `20` | Maximum AST complexity (tokens) permitted by the complexity gate. |
| `opt_tol` | `float` | `1e-5` | Optimization convergence tolerance for L-BFGS-B. |
| `opt_maxiter` | `int` | `50` | Maximum optimization iterations per candidate expression. |
| `num_restarts` | `int` | `5` | Number of random parameter initializations for L-BFGS-B restarts. |

## Scenario Construction Config

When building custom scenarios, you can tune the boundary behavior:

```python
scenario = AnomalyScenario(
    ...
    # Specify the limit behavior for the ARC gate
    limit_variable="x",
    limit_direction="0",   # "0", "inf", or "-inf"
    limit_value=0.0,       # Expected value at boundary (usually 0.0 or a constant)
)
```

## Residual Feature Extraction

The residual analyzer extracts mathematical features from the data to guide the LLM proposers:

- **Monotonicity**: Checks if the residual increases or decreases consistently.
- **Curvature**: Computes the second derivative sign to identify convex/concave shapes.
- **Oscillation**: Checks for zero-crossings to identify periodic/harmonic components.
- **Decay Rate**: Fits an exponential decay to see if the anomaly vanishes at the boundaries.

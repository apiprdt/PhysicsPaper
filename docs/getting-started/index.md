# Getting Started

Welcome to **Anomaly-Driven Correction Discovery (ADCD)**. This guide will help you install ADCD and run your first correction discovery in under 5 minutes.

## Path to Discovery

To get started with ADCD, we recommend following these steps:

1. **[Installation](installation.md)**: Install the library and its scientific dependencies (JAX, SymPy, Matplotlib).
2. **[Quickstart](quickstart.md)**: Learn how to fit a custom experimental dataset in just a few lines of code.
3. **[Your First Discovery](first-discovery.md)**: Explore the built-in benchmark scenarios and see how ADCD uses physics-constrained gates to discover theories from noisy data.

## Why is ADCD different?

Standard symbolic regression attempts to build equations from the ground up, search space size grows exponentially with formula length. ADCD shifts the paradigm:

- **Starts from a known theory**: Takes `y_classical` as a starting point.
- **Learns the correction**: Focuses the symbolic search space exclusively on the discrepancy $\Delta$.
- **Physics-guided**: Employs rigorous physical guardrails (dimensional homogeneity, asymptotic limits) to eliminate nonsense equations before optimization.

# User Guide

This user guide goes deep into the design principles, internal components, and configuration of the Anomaly-Driven Correction Discovery (ADCD) framework.

## Content Navigation

To understand the core mechanisms, explore:

- **[Core Concepts](concepts.md)**: The mathematical foundation of the correction-first paradigm and the L-BFGS-B JAX optimizer.
- **[Anomaly Scenarios](scenarios.md)**: How scenarios are structured, including dynamic ranges, base scales, and variables.
- **[Physics Gates](gates.md)**: A detailed look at the AST complexity, Dimensional, and Asymptotic (ARC) gates.
- **[Proposers](proposers.md)**: Explore the different template generation mechanisms, including the `mock` proposer and LLM-based `hybrid`/`gemini` regimes.
- **[Configuration Options](configuration.md)**: Detailed descriptions of optimizer tolerances, learning budgets, and iteration thresholds.
- **[Export & Visualization](export.md)**: How to extract LaTeX, generate plots, and export raw results for publication.

# API Reference

This section provides the reference documentation for the ADCD public API.

## Core API Functions

- **[`adcd.fit`](fit.md)**: The primary entry point for fitting custom experimental datasets.
- **[`adcd.discover_correction`](discover_correction.md)**: High-level API for running discovery on predefined `Scenario` objects.

## Auxiliary Classes

- **[`adcd.anomaly_scenarios`](scenarios.md)**: Defining and loading benchmark scenarios.
- **[`adcd.result`](result.md)**: The `CorrectionResult` object returned by fitting functions, containing metrics, parameters, and LaTeX export methods.

# Contributing to ADCD

Thank you for your interest in contributing to **Anomaly-Driven Correction Discovery (ADCD)**!

## How to Contribute

### Reporting Bugs
Open a [Bug Report issue](https://github.com/apiprdt/PhysicsPaper/issues/new?template=bug_report.md) with:
- OS and Python version
- Exact command and full traceback
- Minimal reproducible example

### Suggesting Features
Open a [Feature Request issue](https://github.com/apiprdt/PhysicsPaper/issues/new?template=feature_request.md) describing:
- The motivation (what problem does it solve?)
- A sketch of the proposed API or behavior

### Submitting Pull Requests

1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Install** in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Write tests** — all new code must have corresponding unit tests in `tests/`.

4. **Run the test suite** locally before pushing:
   ```bash
   pytest --cov=adcd
   flake8 src/ tests/
   ```

5. **Submit** a Pull Request against `main` and fill out the PR template.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) (enforced by `flake8`, max line length 100).
- Use [Black](https://black.readthedocs.io/) for auto-formatting: `black src/`.
- All public functions and classes must have docstrings.

## Scientific Contributions

If you are adding a new **benchmark scenario**, please:
- Add it to `src/adcd/anomaly_scenarios.py` following the `AnomalyScenario` dataclass pattern.
- Include the classical formula, correction formula, and asymptotic regime definitions.
- Add a unit test in `tests/test_correction_discovery.py` verifying the structural class match.

## License

By contributing, you agree that your contributions will be licensed under the **MIT License**.

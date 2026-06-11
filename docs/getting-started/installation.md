# Installation

ADCD requires Python 3.10 or 3.11.

## Standard Installation

You can install the latest stable version of ADCD from PyPI:

```bash
pip install adcd
```

This will automatically install the core dependencies, including:
- **JAX** (for fast, parameter-scaled L-BFGS-B optimization)
- **SymPy** (for symbolic manipulation, dimensional matching, and physics gates)
- **Matplotlib** (for automatic residual plotting)
- **SciPy** / **NumPy**

## Installation from Source

For development, testing, or reproduction of the paper benchmarks:

```bash
git clone https://github.com/apiprdt/PhysicsPaper.git
cd PhysicsPaper
pip install -e ".[dev]"
```

The `[dev]` extra installs developmental dependencies:
- `pytest` and `pytest-cov` for running the test suite.
- `flake8` and `black` for formatting and linting.
- `mkdocs-material` and `mkdocstrings[python]` if you wish to build this documentation site locally.

## Verification

To verify that the library and its JAX accelerators are functioning correctly on your hardware, run:

```bash
python -c "import adcd; print(adcd.__version__)"
```

To run the full unit test suite (77 tests):

```bash
pytest
```

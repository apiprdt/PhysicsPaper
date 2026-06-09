"""PySR baseline profile definitions (no pysr import — safe for CI unit tests)."""

PYSR_PROFILES = {
    "fast": {
        "niterations": 15,
        "maxsize": 15,
        "timeout_in_seconds": 25,
        "description": "Matched wall-clock budget (legacy comparison)",
    },
    "fair": {
        "niterations": 100,
        "maxsize": 30,
        "timeout_in_seconds": 60,
        "description": "PySR near-default budget (primary fair comparison)",
    },
    "generous": {
        "niterations": 200,
        "maxsize": 40,
        "timeout_in_seconds": 120,
        "description": "Upper-bound PySR search budget",
    },
}

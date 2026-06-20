"""
Regression tests for the multi-seed bootstrap CI aggregator
(scripts/aggregate_seeds.py).

These tests pin the statistical behaviour of the headline "82.8% +/- ... (95% CI)"
number reported in the paper, so a silent regression in the aggregation logic is
caught before tables are regenerated.

The aggregator itself lives outside the package, so we import it via importlib
from its script path (matching how scripts are exercised in CI).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_seeds.py"


@pytest.fixture(scope="module")
def agg():
    """Load scripts/aggregate_seeds.py as a module (it is not on sys.path)."""
    spec = importlib.util.spec_from_file_location("aggregate_seeds", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Synthetic per-trial data used to exercise the aggregator -----------------

def _make_trials(seed_rates, noises=(0.0, 0.01, 0.05, 0.10), n_scenarios=9):
    """Build a fake reproducibility_results.json payload.

    seed_rates: dict seed -> fraction of 36 trials that are class_match=True.
    The matching trials are spread across noise levels so per-noise breakdown
    stays well-defined.
    """
    trials = []
    n_per_noise = n_scenarios
    for seed, rate in seed_rates.items():
        n_match = int(round(rate * n_scenarios * len(noises)))
        # Spread matches across noise levels deterministically.
        flags = [True] * n_match + [False] * (n_scenarios * len(noises) - n_match)
        rng = np.random.default_rng(seed)
        rng.shuffle(flags)
        i = 0
        for noise in noises:
            for _ in range(n_per_noise):
                trials.append({
                    "seed": seed,
                    "noise": noise,
                    "class_match": bool(flags[i]),
                    "tier": "textbook",
                    "nmse_full": 1e-3,
                })
                i += 1
    return trials


class TestSeedRateExtraction:
    def test_per_seed_rate_matches_input(self, agg):
        trials = _make_trials({0: 0.75, 7: 0.944})
        rates = agg._seed_rates(trials)
        assert set(rates) == {0, 7}
        assert rates[0] == pytest.approx(0.75, abs=1e-9)
        # 34/36 = 0.9444..., not exactly 0.944.
        assert rates[7] == pytest.approx(0.94444, abs=1e-4)


class TestBootstrapCI:
    def test_ci_brackets_mean(self, agg):
        # Constant rates => zero variance => degenerate CI at the mean.
        rates = [0.8, 0.8, 0.8, 0.8, 0.8]
        lo, hi = agg.bootstrap_ci(rates, n_bootstrap=1000,
                                  rng=np.random.default_rng(0))
        assert lo == pytest.approx(0.8, abs=1e-9)
        assert hi == pytest.approx(0.8, abs=1e-9)

    def test_ci_contains_sample_mean(self, agg):
        rates = [0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.78]
        mean = float(np.mean(rates))
        lo, hi = agg.bootstrap_ci(rates, n_bootstrap=2000,
                                  rng=np.random.default_rng(1))
        assert lo <= mean + 1e-9
        assert mean - 1e-9 <= hi

    def test_more_variance_gives_wider_ci(self, agg):
        tight = [0.80, 0.82, 0.81, 0.80, 0.79]
        wide = [0.55, 0.95, 0.60, 0.90, 0.50]
        _, hi_tight = agg.bootstrap_ci(tight, n_bootstrap=2000,
                                       rng=np.random.default_rng(2))
        _, hi_wide = agg.bootstrap_ci(wide, n_bootstrap=2000,
                                      rng=np.random.default_rng(2))
        width_tight = hi_tight - np.mean(tight)
        width_wide = hi_wide - np.mean(wide)
        assert width_wide > width_tight

    def test_empty_returns_nan(self, agg):
        lo, hi = agg.bootstrap_ci([], n_bootstrap=100)
        assert np.isnan(lo) and np.isnan(hi)

    def test_reproducible_given_seed(self, agg):
        rates = [0.7, 0.8, 0.9, 0.6, 0.85]
        lo1, hi1 = agg.bootstrap_ci(rates, n_bootstrap=1000,
                                    rng=np.random.default_rng(42))
        lo2, hi2 = agg.bootstrap_ci(rates, n_bootstrap=1000,
                                    rng=np.random.default_rng(42))
        assert (lo1, hi1) == (lo2, hi2)


class TestEndToEndOnExistingResults:
    """If reproducibility_results.json exists, the original 5-seed headline
    number must match the paper-reported mean (82.8%) to guard against silent
    regressions in the aggregation math. We restrict to the original 5 seeds
    (0, 7, 21, 42, 99) so the test is stable while the 20-seed expansion is
    still being appended to the same file."""

    ORIGINAL_SEEDS = [0, 7, 21, 42, 99]

    def test_headline_mean_matches_paper(self, agg):
        results_path = ROOT / "reproducibility_results.json"
        if not results_path.exists():
            pytest.skip("reproducibility_results.json not present")

        all_trials = json.loads(results_path.read_text())
        trials = [r for r in all_trials if r.get("seed") in self.ORIGINAL_SEEDS]
        # Each original seed must have the full 36-trial complement; otherwise
        # the file is mid-rewrite and the test is inconclusive.
        have = sorted(set(r["seed"] for r in trials))
        if have != sorted(self.ORIGINAL_SEEDS):
            pytest.skip("original 5-seed slice not fully present yet")

        rates = agg._seed_rates(trials)
        # Paper: 86.1, 75.0, 77.8, 94.4, 80.6 -> mean 82.8%.
        expected = [0.861, 0.750, 0.778, 0.944, 0.806]
        for s, exp in zip(sorted(self.ORIGINAL_SEEDS), expected):
            assert rates[s] == pytest.approx(exp, abs=2e-3), f"seed {s}"
        mean = float(np.mean(list(rates.values())))
        assert mean == pytest.approx(0.828, abs=2e-3)

        lo, hi = agg.bootstrap_ci(list(rates.values()), n_bootstrap=2000,
                                  rng=np.random.default_rng(0))
        assert lo <= mean <= hi
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

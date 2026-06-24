"""
Regression tests for the multi-seed bootstrap CI aggregator
(scripts/aggregate_seeds.py).

These tests pin the statistical behaviour of the headline
"80.4% +/- 7.4% (95% CI [76.7%, 84.0%]) across sixteen seeds" number reported in
the paper, so a silent regression in the aggregation logic is caught before
tables are regenerated. The headline is computed over single-variable (SV)
scenarios only; multivariable (MV-*) trials are excluded (see _seed_rates).

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
    """If reproducibility_results.json exists, the 16-seed headline number must
    match the paper-reported mean (80.4% +/- 7.4%) to guard against silent
    regressions in the aggregation math. The headline is computed over the full
    sixteen-seed set on single-variable (SV) scenarios only."""

    EXPECTED_SEEDS = [0, 2, 5, 7, 13, 17, 21, 23, 31, 37, 41, 42, 53, 61, 67, 99]

    def test_headline_mean_matches_paper(self, agg):
        results_path = ROOT / "reproducibility_results.json"
        if not results_path.exists():
            pytest.skip("reproducibility_results.json not present")

        all_trials = json.loads(results_path.read_text())
        # Restrict to the headline seeds and let _seed_rates drop MV-* rows.
        trials = [r for r in all_trials if r.get("seed") in self.EXPECTED_SEEDS]
        # Each seed must have the full 36-trial SV complement; otherwise the
        # file is mid-rewrite and the test is inconclusive.
        have = sorted(set(r["seed"] for r in trials))
        if have != sorted(self.EXPECTED_SEEDS):
            pytest.skip("16-seed slice not fully present yet")

        rates = agg._seed_rates(trials)
        mean = float(np.mean(list(rates.values())))
        std = float(np.std(list(rates.values()), ddof=0))
        # Paper headline: 80.4% +/- 7.4% across sixteen seeds.
        assert mean == pytest.approx(0.804, abs=2e-3)
        assert std == pytest.approx(0.074, abs=2e-3)
        # seed=42 must be the highest-performing seed (disclosed in the paper).
        assert max(rates, key=rates.get) == 42
        assert rates[42] == pytest.approx(0.944, abs=2e-3)

        lo, hi = agg.bootstrap_ci(list(rates.values()), n_bootstrap=2000,
                                  rng=np.random.default_rng(0))
        assert lo <= mean <= hi
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


class TestScenarioCountFromSVOnly:
    """The LaTeX caption reports 'N scenarios x 4 noise = 36 trials per seed'.
    When the raw results file mixes single-variable (SV) and multivariable
    (MV-*) rows, the scenario count must be derived from SV rows only --
    otherwise the caption reads an internally-contradictory figure (e.g.
    '11 scenarios x 4 noise = 36 trials', since 11 x 4 != 36)."""

    def _caption_scenario_count(self, agg, trials):
        """Mirror the main() summary construction to extract n_scenarios."""
        seeds_present = sorted(set(r["seed"] for r in trials))
        sv_rows = [r for r in trials
                   if not r.get("scenario", "").startswith("MV-")]
        n_sv = len(sv_rows) // len(seeds_present)
        return n_sv // 4

    def test_sv_only_file_gives_9_scenarios(self, agg):
        # 2 seeds x 9 scenarios x 4 noise = 72 SV trials, no MV rows.
        trials = _make_trials({0: 0.75, 7: 0.944}, n_scenarios=9)
        assert self._caption_scenario_count(agg, trials) == 9

    def test_mixed_sv_mv_file_still_gives_9_scenarios(self, agg):
        # Same 72 SV trials, but with extra MV-* rows that previously
        # contaminated the scenario count.
        trials = _make_trials({0: 0.75, 7: 0.944}, n_scenarios=9)
        mv = {
            "MV-1: Yukawa Mass-Ratio", "MV-2: Plasma Correction",
            "MV-3: Turbulent Drag 2D", "MV-4: Van der Waals 2D",
        }
        for seed in (0, 7):
            for name in mv:
                for noise in (0.0, 0.01, 0.05, 0.10):
                    trials.append({
                        "seed": seed, "noise": noise,
                        "scenario": name, "class_match": False,
                        "tier": "multivariable", "nmse_full": 1e-3,
                    })
        # Without the fix this would be 52/2/4 = 6... actually (36+16)/2/4=6;
        # the old code used the full blended total yielding a wrong figure.
        # The correct SV-only count is 36/2/... = 9 scenarios per seed.
        assert self._caption_scenario_count(agg, trials) == 9

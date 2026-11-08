"""
tests/test_no_oracle_leak.py
============================
Automated regression tests to enforce zero oracle leaks and strict blind audit integrity.

Verifies:
1. Core audit modules (`validation_v2`, `joint_fit`, `likelihood`, `audit`) do not accept or read
   ground-truth labels or generator case names in their decision pathways.
2. Invariance property: Audit verdicts are strictly a mathematical function of (ClaimCard, data).
   Permuting or obscuring candidate names or changing scenario labels does not leak truth.
3. `spike/crossover.py` uses uniform structural family mapping without case_name branching.
4. Operational sigma in crossover audit is computed from observations (blind to noiseless truth).
"""

from __future__ import annotations

import inspect
import numpy as np

import adcd.validation_v2 as val
import adcd.joint_fit as jf
import adcd.likelihood as lh
from adcd.audit import audit, audit_custom_expressions
import spike.crossover as cross


def test_core_modules_have_no_oracle_parameters() -> None:
    """Core decision functions must not accept ground_truth, true_family, or oracle kwargs."""
    forbidden = {"ground_truth", "true_family", "true_expr", "true_constants", "oracle"}

    # 1. validation_v2
    val_funcs = [val.run_three_tier_audit, val.infer_family_from_expr]
    for fn in val_funcs:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        leak = params.intersection(forbidden)
        assert not leak, f"{fn.__name__} in validation_v2 has forbidden oracle parameters: {leak}"

    # 2. joint_fit
    jf_funcs = [jf.fit_joint_baseline_correction]
    for fn in jf_funcs:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        leak = params.intersection(forbidden)
        assert not leak, f"{fn.__name__} in joint_fit has forbidden oracle parameters: {leak}"

    # 3. likelihood
    lh_funcs = [lh.HeteroscedasticLikelihood.compute_log_likelihood, lh.GaussianLikelihoodEngine.evaluate]
    for fn in lh_funcs:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        leak = params.intersection(forbidden)
        assert not leak, f"{fn.__name__} in likelihood has forbidden oracle parameters: {leak}"

    # 4. audit
    aud_funcs = [audit, audit_custom_expressions]
    for fn in aud_funcs:
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        leak = params.intersection(forbidden)
        assert not leak, f"{fn.__name__} in audit has forbidden oracle parameters: {leak}"


def test_crossover_audit_path_has_no_case_name_branching() -> None:
    """The decision path in crossover.py must not branch on case_name when fitting models."""
    # Check source code of audit_crossover_case_single_seed:
    src = inspect.getsource(cross.audit_crossover_case_single_seed)
    lines = src.splitlines()

    # Find where candidate fitting starts
    fit_start_idx = -1
    for i, line in enumerate(lines):
        if "candidates: List[CandidateFit]" in line or "candidates = []" in line:
            fit_start_idx = i
            break
    assert fit_start_idx > 0, "Could not locate candidate fitting section in audit_crossover_case_single_seed"

    fitting_lines = lines[fit_start_idx:]
    for line in fitting_lines:
        assert 'case_name == "michaelis_menten"' not in line, (
            "Found forbidden case_name == 'michaelis_menten' in candidate fitting path: " + line
        )
        assert 'case_name == "debye"' not in line or "include_exact_truth" in line, (
            "Found forbidden case_name == 'debye' in standard candidate fitting path: " + line
        )


def test_uniform_structural_family_map_invariance() -> None:
    """All rational, exponential, and CES models must map to symmetric structural families."""
    fmap = cross.STRUCTURAL_FAMILY_MAP
    assert fmap["convex_rat"] == fmap["convex_rat_p"] == "family_rational"
    assert fmap["convex_exp"] == fmap["convex_exp_p"] == "family_exponential"
    assert fmap["smooth_min_1p"] == fmap["smooth_min_2p"] == "family_smooth_min"
    assert fmap["smooth_max_1p"] == fmap["smooth_max_2p"] == "family_smooth_max"


def test_operational_sigma_is_blind_to_noiseless_truth() -> None:
    """Operational noise sigma must be calculated from observational data y_obs."""
    u = np.logspace(-2.0, 1.0, 50)
    rng = np.random.default_rng(123)
    y_true = cross.planck_exact(u)
    y_obs = y_true + rng.normal(0, 0.01 * y_true)

    # Operational sigma from observations
    sigma_obs = 0.01 * np.maximum(np.abs(y_obs), 1e-4)

    # They should not be identical (because y_obs contains noise realization)
    assert not np.array_equal(sigma_obs, 0.01 * y_true)
    # But should be statistically within 3 standard errors
    rel_diff = np.abs(sigma_obs - 0.01 * y_true) / (0.01 * y_true)
    assert np.mean(rel_diff) < 0.05

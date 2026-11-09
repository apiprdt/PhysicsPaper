"""
tests/test_lagrangian_auditor.py
================================
Unit and regression tests for Lagrangian Field Theory Auditor.
Verifies symbolic and numerical validation of kinetic positivity, gradient
stability, causality bounds, and action reconstructibility.
"""

from __future__ import annotations

import os
import numpy as np

from adcd.lagrangian_auditor import LagrangianFieldAuditor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO_ROOT, "reports", "lagrangian_sparc_coupling.csv")


def test_auditor_detects_ghost_violation() -> None:
    """Verify that a model with negative boost factor is correctly flagged as a ghost."""
    auditor = LagrangianFieldAuditor(n_grid=100)

    # Unphysical model that turns negative at high x
    def unphysical_ghost_fn(x: np.ndarray, s: float) -> np.ndarray:
        return 1.0 - 0.1 * x

    rep = auditor.audit_model("Ghost Model", "y = 1 - 0.1*x", unphysical_ghost_fn)
    assert not rep.no_ghost
    assert not rep.is_physically_sound
    assert any("GHOST" in d for d in rep.diagnostics)


def test_auditor_detects_ellipticity_collapse() -> None:
    """Verify that non-monotonic g_obs (tachyonic/gradient instability) is flagged."""
    auditor = LagrangianFieldAuditor(n_grid=200)

    # Non-monotonic model where g_obs = x*y decreases
    def non_monotonic_fn(x: np.ndarray, s: float) -> np.ndarray:
        return 1.0 / (1.0 + x ** 2)

    rep = auditor.audit_model("Non-Elliptic Model", "y = 1/(1+x^2)", non_monotonic_fn)
    assert not rep.is_elliptic
    assert not rep.is_physically_sound
    assert any("ELLIPTICITY" in d for d in rep.diagnostics)


def test_standard_mond_and_mcgaugh_satisfy_deep_mond_sound_speed_theorem() -> None:
    """
    Verify the theoretical theorem: in the Deep-MOND limit (y ~ s*x^-0.5),
    the radial scalar perturbation sound speed squared approaches exactly c_r^2 = 2.0.
    """
    auditor = LagrangianFieldAuditor(n_grid=1000)

    # McGaugh RAR
    rep_mc = auditor.audit_model(
        "McGaugh RAR",
        "y = 1 / (1 - exp(-sqrt(x/s)))",
        lambda x, s: 1.0 / (-np.expm1(-np.sqrt(x / s))),
        scale_param=0.6513,
    )
    assert rep_mc.no_ghost
    assert rep_mc.is_elliptic
    assert rep_mc.is_physically_sound
    # Max sound speed must be approximately 2.0 (Deep MOND limit)
    assert np.isclose(rep_mc.max_sound_speed_sq, 2.0, atol=0.05)
    assert np.isclose(rep_mc.min_sound_speed_sq, 1.0, atol=0.05)

    # Simple MOND
    rep_sm = auditor.audit_model(
        "Simple MOND",
        "y = (1 + sqrt(1 + 4/(x/s))) / 2",
        lambda x, s: 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (x / s))),
        scale_param=0.6347,
    )
    assert rep_sm.no_ghost
    assert rep_sm.is_elliptic
    assert rep_sm.is_physically_sound
    assert np.isclose(rep_sm.max_sound_speed_sq, 2.0, atol=0.05)
    assert np.isclose(rep_sm.min_sound_speed_sq, 1.0, atol=0.05)


def test_asymptotic_boundary_conditions() -> None:
    """Verify that models correctly recover UV Newtonian limit y -> 1."""
    auditor = LagrangianFieldAuditor(n_grid=500)

    # Model that does not recover Newton (asymptote y -> 2)
    def broken_uv_fn(x: np.ndarray, s: float) -> np.ndarray:
        return 2.0 + 1.0 / np.sqrt(x)

    rep = auditor.audit_model("Broken UV", "y = 2 + x^-0.5", broken_uv_fn)
    assert not rep.asymptotic_uv_newton
    assert not rep.is_physically_sound
    assert any("UV BREAKDOWN" in d for d in rep.diagnostics)


def test_dual_coupling_csv_artifact_validity() -> None:
    """Verify that the generated coupling CSV contains required models and valid columns."""
    assert os.path.exists(CSV_PATH), f"Missing CSV: {CSV_PATH}"
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 7  # Header + 6 models
    header = lines[0].strip().split(",")
    assert "is_physically_sound" in header
    assert "max_c_r2" in header
    assert "bic" in header
    assert "qumond_action" in header

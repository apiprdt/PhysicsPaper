"""
test_imported_phase3.py
=======================
Demonstrates the Blind Three-Tier Protocol on custom imported candidates (e.g. from PySR or literature).
"""

from __future__ import annotations

import numpy as np
from adcd.claim_card import ClaimCard, NoiseSpec
from adcd.audit import audit_custom_expressions


def run_imported_example():
    rng = np.random.RandomState(42)
    N = 150
    r = np.linspace(0.2, 3.5, N)
    
    # Ground truth: Screened Coulomb
    y_cl = 1.0 / r
    delta_true = 0.8 * (np.exp(-r / 1.2) - 1.0)
    sigma_y = 0.01 * y_cl * (1.0 + delta_true)
    y_obs = y_cl * (1.0 + delta_true) + rng.normal(0, sigma_y)

    card = ClaimCard(
        name="Imported Theory Audit: Yukawa Screening",
        variables={"r": "m"},
        baseline_expr="1.0 / r",
        limit_variable="r",
        limit_direction="0",
        window={"r": (0.2, 3.5)},
        mode="multiplicative",
        candidate_expressions=[
            "theta_0 * (exp(-r/theta_1) - 1.0)",
            "theta_0 * (r/theta_1 / (1.0 - r/theta_1))",
            "theta_0 * tanh(r/theta_1)",
        ],
        noise_spec=NoiseSpec(type="column", sigma_col="sigma_y"),
        seed=42,
    )

    data = {
        "r": r,
        "y_obs": y_obs,
        "y_cl": y_cl,
        "sigma_y": sigma_y,
    }

    report = audit_custom_expressions(card, data)
    print("=" * 80)
    print("IMPORTED CANDIDATES AUDIT REPORT:")
    print("=" * 80)
    print(f"Verdict: {report.verdict}")
    print(f"Card Hash: {report.card_hash}")
    print(f"Report Hash: {report.report_hash}")
    print(f"Best Candidate: {report.best_candidate['expr']} (BIC={report.best_candidate['bic']:.2f})")
    print("\nPlain-Language Summary:")
    print(report.plain_language_summary)
    return report


if __name__ == "__main__":
    run_imported_example()

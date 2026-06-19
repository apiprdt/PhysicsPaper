"""MOND interpolating-function comparison for SPARC stacking experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from adcd.metrics import bic_score


def _nmse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_obs - y_pred) ** 2) / max(np.var(y_obs), 1e-15))


SIMPLE_MOND_FORMULA = "(1 + sqrt(1 + 4/x)) / 2"
STANDARD_MOND_FORMULA = "1 / sqrt(1 - exp(-sqrt(x)))"
RAR_FORMULA = "1 / (1 - exp(-sqrt(x)))"


def nu_simple_mond(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (1.0 + np.sqrt(1.0 + 4.0 / x)) / 2.0


def nu_standard_mond(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / np.sqrt(1.0 - np.exp(-np.sqrt(x)))


def nu_rar(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


@dataclass
class MondModelScore:
    name: str
    formula: str
    nmse: float
    bic: float
    n_params: int


def score_mond_models(
    x: np.ndarray,
    nu_obs: np.ndarray,
    adcd_nmse: float | None = None,
    adcd_n_params: int = 2,
) -> List[MondModelScore]:
    """Compare fixed MOND variants (+ optional ADCD) on stacked ν(x) data."""
    x = np.asarray(x, dtype=float)
    nu_obs = np.asarray(nu_obs, dtype=float)
    n = len(nu_obs)

    scores: List[MondModelScore] = []
    for name, formula, pred in [
        ("Simple MOND", SIMPLE_MOND_FORMULA, nu_simple_mond(x)),
        ("Standard MOND", STANDARD_MOND_FORMULA, nu_standard_mond(x)),
        ("RAR (McGaugh)", RAR_FORMULA, nu_rar(x)),
    ]:
        mse = float(np.mean((nu_obs - pred) ** 2))
        nmse_val = _nmse(nu_obs, pred)
        scores.append(MondModelScore(
            name=name, formula=formula, nmse=nmse_val,
            bic=bic_score(nmse_val, 0, n), n_params=0,
        ))

    if adcd_nmse is not None:
        scores.append(MondModelScore(
            name="ADCD discovered",
            formula="(from iADCD search)",
            nmse=adcd_nmse,
            bic=bic_score(adcd_nmse, adcd_n_params, n),
            n_params=adcd_n_params,
        ))

    scores.sort(key=lambda s: s.bic)
    return scores


def print_mond_comparison(scores: List[MondModelScore]) -> None:
    print("\n=== MOND Model Comparison (lower BIC = better) ===")
    print(f"{'Rank':>4} | {'Model':<18} | {'NMSE':>10} | {'BIC':>10} | {'k':>3}")
    for i, s in enumerate(scores, 1):
        print(f"{i:>4} | {s.name:<18} | {s.nmse:>10.5f} | {s.bic:>10.2f} | {s.n_params:>3}")


def scores_to_dict(scores: List[MondModelScore]) -> List[Dict]:
    return [
        {"name": s.name, "formula": s.formula, "nmse": s.nmse, "bic": s.bic, "n_params": s.n_params}
        for s in scores
    ]

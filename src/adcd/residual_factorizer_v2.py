"""Variance-decomposition residual factorizer for multivariable corrections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

FactorizationType = Literal["multiplicative", "additive", "none"]


@dataclass
class FactorizationResult:
    factorization_type: FactorizationType
    explained_variance: float
    components: Dict[str, np.ndarray] = field(default_factory=dict)


class ResidualFactorizerV2:
    """
    Test multiplicative / additive separability via variance decomposition.

    Multiplicative: fit log|δ| ≈ s₁(x₁) + s₂(x₂) in spline basis.
    Additive: fit δ ≈ s₁(x₁) + s₂(x₂) directly.
    """

    def test_separability(
        self,
        X: Dict[str, np.ndarray],
        delta: np.ndarray,
        r2_threshold: float = 0.85,
    ) -> FactorizationResult:
        var_names = list(X.keys())
        if len(var_names) < 2:
            return FactorizationResult(
                factorization_type="none",
                explained_variance=0.0,
            )

        X_arr = np.column_stack([X[v] for v in var_names])
        spline = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        X_spline = spline.fit_transform(X_arr)

        log_delta = np.log(np.abs(delta) + 1e-15)
        model_mult = Ridge(alpha=0.01)
        model_mult.fit(X_spline, log_delta)
        r2_mult = float(model_mult.score(X_spline, log_delta))

        model_add = Ridge(alpha=0.01)
        model_add.fit(X_spline, delta)
        r2_add = float(model_add.score(X_spline, delta))

        if r2_mult >= r2_threshold and r2_mult >= r2_add:
            return FactorizationResult(
                factorization_type="multiplicative",
                explained_variance=r2_mult,
            )
        if r2_add >= r2_threshold:
            return FactorizationResult(
                factorization_type="additive",
                explained_variance=r2_add,
            )

        return FactorizationResult(
            factorization_type="none",
            explained_variance=max(r2_mult, r2_add),
        )

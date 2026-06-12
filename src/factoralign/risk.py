from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RiskModel:
    """Linear factor risk model.

    Parameters
    ----------
    X:
        Asset-by-factor exposure matrix.
    factor_cov:
        Factor covariance matrix.
    specific_var:
        Asset-level specific variances.
    asset_ids:
        Optional asset identifiers.
    factor_names:
        Optional factor names.
    """

    X: np.ndarray
    factor_cov: np.ndarray
    specific_var: np.ndarray
    asset_ids: Sequence[str] | None = None
    factor_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        factor_cov = np.asarray(self.factor_cov, dtype=float)
        specific_var = np.asarray(self.specific_var, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_assets, n_factors).")

        n_assets, n_factors = X.shape

        if factor_cov.shape != (n_factors, n_factors):
            raise ValueError(
                "factor_cov must have shape (n_factors, n_factors). "
                f"Expected {(n_factors, n_factors)}, got {factor_cov.shape}."
            )

        if specific_var.shape != (n_assets,):
            raise ValueError(
                "specific_var must be a 1D array with length n_assets. "
                f"Expected {(n_assets,)}, got {specific_var.shape}."
            )

        if np.any(specific_var <= 0):
            raise ValueError("All specific variances must be strictly positive.")

        if self.asset_ids is not None and len(self.asset_ids) != n_assets:
            raise ValueError("asset_ids must have length n_assets.")

        if self.factor_names is not None and len(self.factor_names) != n_factors:
            raise ValueError("factor_names must have length n_factors.")

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "factor_cov", factor_cov)
        object.__setattr__(self, "specific_var", specific_var)

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.X.shape[0]

    @property
    def n_factors(self) -> int:
        """Number of risk-model factors."""
        return self.X.shape[1]

    def covariance(self) -> np.ndarray:
        """Return the full asset covariance matrix Q."""
        return self.X @ self.factor_cov @ self.X.T + np.diag(self.specific_var)

    def inverse_specific_var_weights(self) -> np.ndarray:
        """Return WLS weights equal to inverse specific variance."""
        return 1.0 / self.specific_var
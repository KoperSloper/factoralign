from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    """Result of projecting an asset-level vector onto risk-model factors."""

    original: np.ndarray
    spanned: np.ndarray
    orthogonal: np.ndarray
    coefficients: np.ndarray


    def orthogonality_error(
        self,
        exposures: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Return ||X.T W orthogonal||."""
        X = _as_2d_array(exposures, "exposures")
        w = _validate_weights(weights, X.shape[0])

        return float(np.linalg.norm(X.T @ (w * self.orthogonal)))


def project_onto_factors(
    vector: np.ndarray,
    exposures: np.ndarray,
    weights: np.ndarray | None = None,
    rcond: float | None = None,
) -> ProjectionResult:
    """Decompose an asset-level vector into spanned and orthogonal components.

    The decomposition is

        vector = spanned + orthogonal

    where `spanned` lies in the column space of `exposures`, and `orthogonal`
    satisfies

        X.T W orthogonal = 0.

    If `weights` is None, ordinary least squares is used. If `weights` is
    provided, weighted least squares is used. In the factor-alignment workflow,
    a natural default is inverse specific variance, i.e. 1 / specific_var.

    Parameters
    ----------
    vector:
        Asset-level vector with shape (n_assets,).
    exposures:
        Asset-by-factor exposure matrix X with shape (n_assets, n_factors).
    weights:
        Optional positive WLS weights with shape (n_assets,).
    rcond:
        Cutoff passed to np.linalg.lstsq.

    Returns
    -------
    ProjectionResult
        Original vector, spanned component, orthogonal component, and factor
        coefficients.
    """
    z = _as_1d_array(vector, "vector")
    X = _as_2d_array(exposures, "exposures")

    n_assets, _ = X.shape

    if z.shape != (n_assets,):
        raise ValueError(f"vector must have shape {(n_assets,)}, got {z.shape}.")

    w = _validate_weights(weights, n_assets)

    # Weighted least squares:
    #   min_b sum_i w_i (z_i - X_i b)^2
    #
    # Equivalent OLS problem:
    #   min_b ||sqrt(W) z - sqrt(W) X b||^2
    sqrt_w = np.sqrt(w)
    X_weighted = X * sqrt_w[:, None]
    z_weighted = z * sqrt_w

    coefficients, *_ = np.linalg.lstsq(X_weighted, z_weighted, rcond=rcond)

    spanned = X @ coefficients
    orthogonal = z - spanned

    return ProjectionResult(
        original=z,
        spanned=spanned,
        orthogonal=orthogonal,
        coefficients=coefficients,
    )


def normalize_vector(vector: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return vector / ||vector||.

    Raises an error if the vector norm is numerically zero.
    """
    z = _as_1d_array(vector, "vector")
    norm = np.linalg.norm(z)

    if norm <= tol:
        raise ValueError("Cannot normalize a vector with near-zero norm.")

    return z / norm


def _as_1d_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    return array


def _as_2d_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    return array


def _validate_weights(weights: np.ndarray | None, n_assets: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_assets)

    w = _as_1d_array(weights, "weights")

    if w.shape != (n_assets,):
        raise ValueError(f"weights must have shape {(n_assets,)}, got {w.shape}.")

    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive.")

    return w
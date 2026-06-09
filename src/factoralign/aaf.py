from __future__ import annotations

import numpy as np

from .constraints import LinearConstraints
from .optimizer import MVOResult, solve_mvo
from .projection import normalize_vector


def augment_covariance(
    covariance: np.ndarray,
    latent_factor: np.ndarray,
    latent_variance: float,
    normalize: bool = True,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return the AAF-augmented covariance matrix.

    The augmented covariance is

        Q_aug = Q + nu * y y.T

    where y is the latent factor exposure vector and nu is the estimated
    variance of its factor return.

    Parameters
    ----------
    covariance:
        Base asset covariance matrix Q.
    latent_factor:
        Latent factor exposure vector y.
    latent_variance:
        Estimated latent factor variance nu.
    normalize:
        If True, normalize latent_factor before constructing y y.T.
    tol:
        Numerical tolerance used when normalizing.
    """
    Q = _as_square_matrix(covariance, "covariance")
    y = _as_1d_array(latent_factor, "latent_factor")

    if y.shape != (Q.shape[0],):
        raise ValueError(
            f"latent_factor must have shape {(Q.shape[0],)}, got {y.shape}."
        )

    if latent_variance < 0:
        raise ValueError("latent_variance must be non-negative.")

    if latent_variance == 0:
        return 0.5 * (Q + Q.T)

    if normalize:
        y = normalize_vector(y, tol=tol)

    Q_aug = Q + latent_variance * np.outer(y, y)

    return 0.5 * (Q_aug + Q_aug.T)


def aaf_variance_contribution(
    holdings: np.ndarray,
    latent_factor: np.ndarray,
    latent_variance: float,
    normalize: bool = True,
    tol: float = 1e-12,
) -> float:
    """Return the extra AAF variance term nu * (h.T y)^2."""
    h = _as_1d_array(holdings, "holdings")
    y = _as_1d_array(latent_factor, "latent_factor")

    if h.shape != y.shape:
        raise ValueError(f"holdings and latent_factor must have the same shape.")

    if latent_variance < 0:
        raise ValueError("latent_variance must be non-negative.")

    if normalize:
        y = normalize_vector(y, tol=tol)

    exposure = float(h @ y)

    return float(latent_variance * exposure**2)


def solve_aaf_mvo(
    alpha: np.ndarray,
    covariance: np.ndarray,
    constraints: LinearConstraints,
    risk_aversion: float,
    latent_factor: np.ndarray,
    latent_variance: float,
    normalize: bool = True,
    solver: str | None = None,
) -> MVOResult:
    """Solve MVO using the AAF-augmented covariance matrix.

    This solves the same portfolio problem as `solve_mvo`, but replaces Q with

        Q_aug = Q + nu * y y.T.
    """
    Q_aug = augment_covariance(
        covariance=covariance,
        latent_factor=latent_factor,
        latent_variance=latent_variance,
        normalize=normalize,
    )

    return solve_mvo(
        alpha=alpha,
        covariance=Q_aug,
        constraints=constraints,
        risk_aversion=risk_aversion,
        solver=solver,
    )


def _as_1d_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    return array


def _as_square_matrix(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square 2D array.")

    return array
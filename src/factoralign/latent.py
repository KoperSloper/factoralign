from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatentFactorReturn:
    """One-period latent factor return estimate."""

    latent_return: float
    regular_returns: np.ndarray
    residuals: np.ndarray


@dataclass(frozen=True)
class LatentFactorReturnSeries:
    """Time series of latent factor return estimates."""

    latent_returns: np.ndarray
    regular_returns: np.ndarray
    residuals: np.ndarray


def estimate_latent_factor_return(
    returns: np.ndarray,
    exposures: np.ndarray,
    latent_factor: np.ndarray,
    weights: np.ndarray | None = None,
    rcond: float | None = None,
) -> LatentFactorReturn:
    """Estimate one realized latent factor return.

    Runs the cross-sectional regression

        r = X f + y f_y + eps

    where
        r   is the realized asset return vector,
        X   is the regular risk-model exposure matrix,
        y   is the latent factor exposure vector,
        f_y is the latent factor return.

    If weights are supplied, weighted least squares is used.
    """
    r = _as_1d_array(returns, "returns")
    X = _as_2d_array(exposures, "exposures")
    y = _as_1d_array(latent_factor, "latent_factor")

    n_assets = r.size

    if X.shape[0] != n_assets:
        raise ValueError(
            f"exposures must have {n_assets} rows, got {X.shape[0]}."
        )

    if y.shape != (n_assets,):
        raise ValueError(
            f"latent_factor must have shape {(n_assets,)}, got {y.shape}."
        )

    if np.linalg.norm(y) == 0:
        raise ValueError("latent_factor must not be the zero vector.")

    w = _validate_weights(weights, n_assets)

    design = np.column_stack([X, y])

    sqrt_w = np.sqrt(w)
    design_weighted = design * sqrt_w[:, None]
    returns_weighted = r * sqrt_w

    coefficients, *_ = np.linalg.lstsq(
        design_weighted,
        returns_weighted,
        rcond=rcond,
    )

    fitted = design @ coefficients
    residuals = r - fitted

    return LatentFactorReturn(
        latent_return=float(coefficients[-1]),
        regular_returns=coefficients[:-1],
        residuals=residuals,
    )


def estimate_latent_factor_returns(
    returns: np.ndarray,
    exposures: np.ndarray,
    latent_factors: np.ndarray,
    weights: np.ndarray | None = None,
    rcond: float | None = None,
) -> LatentFactorReturnSeries:
    """Estimate a time series of latent factor returns.

    Row t must contain:

        returns[t]        = realized asset returns from t to t+1
        exposures[t]      = risk exposures observed at time t
        latent_factors[t] = latent factor exposure y_t observed at time t
        weights[t]        = optional regression weights observed at time t

    The function does not shift arrays internally. The caller is responsible
    for passing already-aligned inputs.

    Parameters
    ----------
    returns:
        Realized asset returns with shape (n_periods, n_assets).
    exposures:
        Either fixed exposures with shape (n_assets, n_factors), or
        time-varying exposures with shape (n_periods, n_assets, n_factors).
    latent_factors:
        Latent factor exposures with shape (n_periods, n_assets).
    weights:
        Optional fixed weights with shape (n_assets,), or time-varying weights
        with shape (n_periods, n_assets).
    """
    R = _as_2d_array(returns, "returns")
    Y = _as_2d_array(latent_factors, "latent_factors")

    n_periods, n_assets = R.shape

    if Y.shape != (n_periods, n_assets):
        raise ValueError(
            f"latent_factors must have shape {(n_periods, n_assets)}, got {Y.shape}."
        )

    X_series = _prepare_exposures(exposures, n_periods, n_assets)
    W_series = _prepare_weights(weights, n_periods, n_assets)

    latent_returns = np.empty(n_periods)
    regular_returns = []
    residuals = np.empty_like(R)

    for t in range(n_periods):
        result = estimate_latent_factor_return(
            returns=R[t],
            exposures=X_series[t],
            latent_factor=Y[t],
            weights=W_series[t],
            rcond=rcond,
        )

        latent_returns[t] = result.latent_return
        regular_returns.append(result.regular_returns)
        residuals[t] = result.residuals

    return LatentFactorReturnSeries(
        latent_returns=latent_returns,
        regular_returns=np.vstack(regular_returns),
        residuals=residuals,
    )


def estimate_latent_variance(
    latent_returns: np.ndarray,
    ddof: int = 1,
    annualization: float = 1.0,
) -> float:
    """Estimate latent factor variance ν from a return series.

    By default this returns the sample variance using ddof=1.
    Set annualization=12 for monthly returns, 252 for daily returns, etc.
    """
    f = _as_1d_array(latent_returns, "latent_returns")

    if f.size <= ddof:
        raise ValueError("Not enough observations to estimate variance.")

    return float(annualization * np.var(f, ddof=ddof))


def _prepare_exposures(
    exposures: np.ndarray,
    n_periods: int,
    n_assets: int,
) -> np.ndarray:
    X = np.asarray(exposures, dtype=float)

    if X.ndim == 2:
        if X.shape[0] != n_assets:
            raise ValueError(
                f"exposures must have {n_assets} rows, got {X.shape[0]}."
            )
        return np.repeat(X[None, :, :], n_periods, axis=0)

    if X.ndim == 3:
        if X.shape[:2] != (n_periods, n_assets):
            raise ValueError(
                "time-varying exposures must have shape "
                f"{(n_periods, n_assets, 'n_factors')}, got {X.shape}."
            )
        return X

    raise ValueError("exposures must be either 2D or 3D.")


def _prepare_weights(
    weights: np.ndarray | None,
    n_periods: int,
    n_assets: int,
) -> np.ndarray:
    if weights is None:
        return np.ones((n_periods, n_assets))

    w = np.asarray(weights, dtype=float)

    if w.ndim == 1:
        _validate_weights(w, n_assets)
        return np.repeat(w[None, :], n_periods, axis=0)

    if w.ndim == 2:
        if w.shape != (n_periods, n_assets):
            raise ValueError(
                f"weights must have shape {(n_periods, n_assets)}, got {w.shape}."
            )
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive.")
        return w

    raise ValueError("weights must be either 1D or 2D.")


def _validate_weights(weights: np.ndarray | None, n_assets: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_assets)

    w = _as_1d_array(weights, "weights")

    if w.shape != (n_assets,):
        raise ValueError(f"weights must have shape {(n_assets,)}, got {w.shape}.")

    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive.")

    return w


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
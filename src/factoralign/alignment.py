from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import LinearConstraints
from .projection import ProjectionResult, normalize_vector, project_onto_factors


@dataclass(frozen=True)
class ImpliedAlpha:
    """Constraint pressure and implied alpha."""

    pressure_ineq: np.ndarray
    pressure_eq: np.ndarray
    pressure_all: np.ndarray
    gamma_ineq: np.ndarray
    gamma_all: np.ndarray


@dataclass(frozen=True)
class AlignmentAnalysis:
    """Projection-based alignment analysis."""

    implied: ImpliedAlpha
    alpha: ProjectionResult
    pressure_ineq: ProjectionResult
    pressure_all: ProjectionResult
    gamma_ineq: ProjectionResult
    gamma_all: ProjectionResult

    def gamma_direction(
        self,
        include_equalities: bool = False,
        tol: float = 1e-12,
    ) -> np.ndarray:
        """Return normalized orthogonal implied-alpha direction."""
        projection = self.gamma_all if include_equalities else self.gamma_ineq
        return normalize_vector(projection.orthogonal, tol=tol)


def compute_implied_alpha(
    alpha: np.ndarray,
    constraints: LinearConstraints,
    ineq_duals: np.ndarray,
    eq_duals: np.ndarray | None = None,
) -> ImpliedAlpha:
    """Compute constraint pressure and implied alpha.

    Uses the convention

        gamma_all = alpha - G.T @ mu - C.T @ nu

    and

        gamma_ineq = alpha - G.T @ mu.

    Parameters
    ----------
    alpha:
        Asset-level alpha vector.
    constraints:
        Linear constraints in canonical form G h <= u, C h = d.
    ineq_duals:
        Dual variables mu for inequality constraints.
    eq_duals:
        Dual variables nu for equality constraints. If None, treated as zero.
    """
    alpha = _as_1d_array(alpha, "alpha")
    mu = _as_1d_array(ineq_duals, "ineq_duals")

    if alpha.shape != (constraints.n_assets,):
        raise ValueError(
            f"alpha must have shape {(constraints.n_assets,)}, got {alpha.shape}."
        )

    if mu.shape != (constraints.n_ineq,):
        raise ValueError(
            f"ineq_duals must have shape {(constraints.n_ineq,)}, got {mu.shape}."
        )

    if eq_duals is None:
        nu = np.zeros(constraints.n_eq)
    else:
        nu = _as_1d_array(eq_duals, "eq_duals")

    if nu.shape != (constraints.n_eq,):
        raise ValueError(
            f"eq_duals must have shape {(constraints.n_eq,)}, got {nu.shape}."
        )

    pressure_ineq = constraints.G.T @ mu
    pressure_eq = constraints.C.T @ nu
    pressure_all = pressure_ineq + pressure_eq

    gamma_ineq = alpha - pressure_ineq
    gamma_all = alpha - pressure_all

    return ImpliedAlpha(
        pressure_ineq=pressure_ineq,
        pressure_eq=pressure_eq,
        pressure_all=pressure_all,
        gamma_ineq=gamma_ineq,
        gamma_all=gamma_all,
    )


def analyze_alignment(
    alpha: np.ndarray,
    constraints: LinearConstraints,
    ineq_duals: np.ndarray,
    exposures: np.ndarray,
    weights: np.ndarray | None = None,
    eq_duals: np.ndarray | None = None,
    rcond: float | None = None,
) -> AlignmentAnalysis:
    """Compute implied alpha and decompose key vectors against risk factors.

    Projects the following vectors onto the column space of `exposures`:

        alpha
        G.T @ mu
        G.T @ mu + C.T @ nu
        alpha - G.T @ mu
        alpha - G.T @ mu - C.T @ nu

    The orthogonal components are the parts not explained by the risk model.
    """
    implied = compute_implied_alpha(
        alpha=alpha,
        constraints=constraints,
        ineq_duals=ineq_duals,
        eq_duals=eq_duals,
    )

    alpha_projection = project_onto_factors(
        vector=alpha,
        exposures=exposures,
        weights=weights,
        rcond=rcond,
    )

    pressure_ineq_projection = project_onto_factors(
        vector=implied.pressure_ineq,
        exposures=exposures,
        weights=weights,
        rcond=rcond,
    )

    pressure_all_projection = project_onto_factors(
        vector=implied.pressure_all,
        exposures=exposures,
        weights=weights,
        rcond=rcond,
    )

    gamma_ineq_projection = project_onto_factors(
        vector=implied.gamma_ineq,
        exposures=exposures,
        weights=weights,
        rcond=rcond,
    )

    gamma_all_projection = project_onto_factors(
        vector=implied.gamma_all,
        exposures=exposures,
        weights=weights,
        rcond=rcond,
    )

    return AlignmentAnalysis(
        implied=implied,
        alpha=alpha_projection,
        pressure_ineq=pressure_ineq_projection,
        pressure_all=pressure_all_projection,
        gamma_ineq=gamma_ineq_projection,
        gamma_all=gamma_all_projection,
    )


def orthogonal_norm_share(projection: ProjectionResult) -> float:
    """Return ||z_orthogonal|| / ||z||."""
    original_norm = np.linalg.norm(projection.original)

    if original_norm == 0:
        return 0.0

    return float(np.linalg.norm(projection.orthogonal) / original_norm)


def _as_1d_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    return array
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .constraints import LinearConstraints


@dataclass(frozen=True)
class MVOResult:
    """Output of a mean-variance optimization."""

    holdings: np.ndarray
    ineq_duals: np.ndarray
    eq_duals: np.ndarray
    status: str
    objective_value: float


def solve_mvo(
    alpha: np.ndarray,
    covariance: np.ndarray,
    constraints: LinearConstraints,
    risk_aversion: float,
    solver: str | None = None,
) -> MVOResult:
    """Solve constrained mean-variance optimization.

    Solves:

        max_h alpha.T h - 0.5 * risk_aversion * h.T Q h
        s.t.  G h <= u
              C h = d
    """
    alpha = np.asarray(alpha, dtype=float)
    Q = np.asarray(covariance, dtype=float)

    if alpha.ndim != 1:
        raise ValueError("alpha must be a 1D array.")

    if Q.shape != (alpha.size, alpha.size):
        raise ValueError("covariance must have shape (n_assets, n_assets).")

    if constraints.n_assets != alpha.size:
        raise ValueError("constraints and alpha have inconsistent dimensions.")

    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")

    n_assets = alpha.size
    h = cp.Variable(n_assets)

    objective = cp.Maximize(
        alpha @ h - 0.5 * risk_aversion * cp.quad_form(h, cp.psd_wrap(Q))
    )

    cvx_constraints = []
    ineq_constraint = None
    eq_constraint = None

    if constraints.n_ineq > 0:
        ineq_constraint = constraints.G @ h <= constraints.u
        cvx_constraints.append(ineq_constraint)

    if constraints.n_eq > 0:
        eq_constraint = constraints.C @ h == constraints.d
        cvx_constraints.append(eq_constraint)

    problem = cp.Problem(objective, cvx_constraints)

    if solver is None:
        problem.solve()
    else:
        problem.solve(solver=solver)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")

    ineq_duals = (
        np.asarray(ineq_constraint.dual_value, dtype=float)
        if ineq_constraint is not None
        else np.zeros(0)
    )

    eq_duals = (
        np.asarray(eq_constraint.dual_value, dtype=float)
        if eq_constraint is not None
        else np.zeros(0)
    )

    return MVOResult(
        holdings=np.asarray(h.value, dtype=float),
        ineq_duals=ineq_duals,
        eq_duals=eq_duals,
        status=problem.status,
        objective_value=float(problem.value), # type: ignore
    )


def kkt_residual(
    alpha: np.ndarray,
    covariance: np.ndarray,
    constraints: LinearConstraints,
    result: MVOResult,
    risk_aversion: float,
) -> np.ndarray:
    """Return λQh - α + G.T μ + C.T ν."""
    return (
        risk_aversion * covariance @ result.holdings
        - alpha
        + constraints.G.T @ result.ineq_duals
        + constraints.C.T @ result.eq_duals
    )
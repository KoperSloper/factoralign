from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LinearConstraints:
    """Canonical linear constraints for portfolio optimization.

    Parameters
    ----------
    G:
        Inequality constraint matrix, shape (n_ineq, n_assets).
    u:
        Inequality upper-bound vector, shape (n_ineq,).
    C:
        Equality constraint matrix, shape (n_eq, n_assets).
    d:
        Equality target vector, shape (n_eq,).
    names:
        Optional names for inequality constraints.
    groups:
        Optional group labels for inequality constraints.
    eq_names:
        Optional names for equality constraints.
    """

    G: np.ndarray
    u: np.ndarray
    C: np.ndarray
    d: np.ndarray
    names: Sequence[str] | None = None
    groups: Sequence[str] | None = None
    eq_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        G = np.asarray(self.G, dtype=float)
        u = np.asarray(self.u, dtype=float)
        C = np.asarray(self.C, dtype=float)
        d = np.asarray(self.d, dtype=float)

        if G.ndim != 2:
            raise ValueError("G must be a 2D array.")
        if C.ndim != 2:
            raise ValueError("C must be a 2D array.")

        n_ineq, n_assets = G.shape
        n_eq, n_assets_eq = C.shape

        if n_assets != n_assets_eq:
            raise ValueError("G and C must have the same number of columns.")

        if u.shape != (n_ineq,):
            raise ValueError(f"u must have shape {(n_ineq,)}, got {u.shape}.")

        if d.shape != (n_eq,):
            raise ValueError(f"d must have shape {(n_eq,)}, got {d.shape}.")

        if self.names is not None and len(self.names) != n_ineq:
            raise ValueError("names must have length equal to number of inequalities.")

        if self.groups is not None and len(self.groups) != n_ineq:
            raise ValueError("groups must have length equal to number of inequalities.")

        if self.eq_names is not None and len(self.eq_names) != n_eq:
            raise ValueError("eq_names must have length equal to number of equalities.")

        object.__setattr__(self, "G", G)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "d", d)

    @property
    def n_assets(self) -> int:
        return self.G.shape[1]

    @property
    def n_ineq(self) -> int:
        return self.G.shape[0]

    @property
    def n_eq(self) -> int:
        return self.C.shape[0]

    @classmethod
    def empty(cls, n_assets: int) -> "LinearConstraints":
        """Create an empty constraint object."""
        return cls(
            G=np.zeros((0, n_assets)),
            u=np.zeros(0),
            C=np.zeros((0, n_assets)),
            d=np.zeros(0),
            names=[],
            groups=[],
            eq_names=[],
        )

    def slack(self, h: np.ndarray) -> np.ndarray:
        """Return inequality slack u - G h."""
        h = np.asarray(h, dtype=float)

        if h.shape != (self.n_assets,):
            raise ValueError(f"h must have shape {(self.n_assets,)}, got {h.shape}.")

        return self.u - self.G @ h

    def binding_mask(self, h: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Return boolean mask for approximately binding inequalities."""
        return self.slack(h) <= tol

    def combine(self, other: "LinearConstraints") -> "LinearConstraints":
        """Combine two constraint objects with the same number of assets."""
        if self.n_assets != other.n_assets:
            raise ValueError("Cannot combine constraints with different n_assets.")

        names = list(self.names or default_ineq_names(self.n_ineq, "ineq"))
        names += list(other.names or default_ineq_names(other.n_ineq, "ineq"))

        groups = list(self.groups or ["unknown"] * self.n_ineq)
        groups += list(other.groups or ["unknown"] * other.n_ineq)

        eq_names = list(self.eq_names or default_eq_names(self.n_eq, "eq"))
        eq_names += list(other.eq_names or default_eq_names(other.n_eq, "eq"))

        return LinearConstraints(
            G=np.vstack([self.G, other.G]),
            u=np.concatenate([self.u, other.u]),
            C=np.vstack([self.C, other.C]),
            d=np.concatenate([self.d, other.d]),
            names=names,
            groups=groups,
            eq_names=eq_names,
        )


def default_ineq_names(n: int, prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def default_eq_names(n: int, prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def fully_invested(n_assets: int, total: float = 1.0) -> LinearConstraints:
    """Create the equality constraint 1.T h = total."""
    C = np.ones((1, n_assets))
    d = np.array([total], dtype=float)

    return LinearConstraints(
        G=np.zeros((0, n_assets)),
        u=np.zeros(0),
        C=C,
        d=d,
        names=[],
        groups=[],
        eq_names=["fully_invested"],
    )


def long_only(n_assets: int) -> LinearConstraints:
    """Create long-only constraints h >= 0, represented as -h <= 0."""
    G = -np.eye(n_assets)
    u = np.zeros(n_assets)

    return LinearConstraints(
        G=G,
        u=u,
        C=np.zeros((0, n_assets)),
        d=np.zeros(0),
        names=[f"long_only_{i}" for i in range(n_assets)],
        groups=["long_only"] * n_assets,
        eq_names=[],
    )


def box_bounds(
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    n_assets: int | None = None,
) -> LinearConstraints:
    """Create simple holding bounds lower <= h <= upper.

    Bounds may be scalars or asset-level vectors.
    """
    lower_vec, upper_vec = _expand_bounds(lower, upper, n_assets)

    n_assets = lower_vec.size

    # h <= upper
    G_upper = np.eye(n_assets)
    u_upper = upper_vec

    # h >= lower  <=>  -h <= -lower
    G_lower = -np.eye(n_assets)
    u_lower = -lower_vec

    G = np.vstack([G_upper, G_lower])
    u = np.concatenate([u_upper, u_lower])

    return LinearConstraints(
        G=G,
        u=u,
        C=np.zeros((0, n_assets)),
        d=np.zeros(0),
        names=[f"upper_bound_{i}" for i in range(n_assets)]
        + [f"lower_bound_{i}" for i in range(n_assets)],
        groups=["upper_bound"] * n_assets + ["lower_bound"] * n_assets,
        eq_names=[],
    )


def active_bounds(
    benchmark: np.ndarray,
    lower_active: float | np.ndarray,
    upper_active: float | np.ndarray,
) -> LinearConstraints:
    """Create active holding bounds lower <= h - b <= upper.

    Parameters
    ----------
    benchmark:
        Benchmark weights b.
    lower_active:
        Lower active bound l. May be scalar or vector.
    upper_active:
        Upper active bound u. May be scalar or vector.
    """
    benchmark = np.asarray(benchmark, dtype=float)
    n_assets = benchmark.size

    lower_vec, upper_vec = _expand_bounds(
        lower_active, upper_active, n_assets=n_assets
    )

    # h - b <= upper  <=>  h <= b + upper
    G_upper = np.eye(n_assets)
    u_upper = benchmark + upper_vec

    # h - b >= lower  <=>  -h <= -lower - b
    G_lower = -np.eye(n_assets)
    u_lower = -lower_vec - benchmark

    G = np.vstack([G_upper, G_lower])
    u = np.concatenate([u_upper, u_lower])

    return LinearConstraints(
        G=G,
        u=u,
        C=np.zeros((0, n_assets)),
        d=np.zeros(0),
        names=[f"active_upper_{i}" for i in range(n_assets)]
        + [f"active_lower_{i}" for i in range(n_assets)],
        groups=["active_upper"] * n_assets + ["active_lower"] * n_assets,
        eq_names=[],
    )


def factor_active_bounds(
    exposures: np.ndarray,
    benchmark: np.ndarray,
    lower_active: float | np.ndarray,
    upper_active: float | np.ndarray,
    factor_names: Sequence[str] | None = None,
    group: str = "factor_active_bound",
) -> LinearConstraints:
    """Create active factor exposure bounds.

    The constraint is

        lower <= X.T @ (h - b) <= upper

    where `exposures` is X with shape (n_assets, n_factors).
    """
    X = np.asarray(exposures, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    if X.ndim != 2:
        raise ValueError("exposures must be a 2D array.")

    n_assets, n_factors = X.shape

    if benchmark.shape != (n_assets,):
        raise ValueError(
            f"benchmark must have shape {(n_assets,)}, got {benchmark.shape}."
        )

    lower_vec, upper_vec = _expand_bounds(
        lower_active, upper_active, n_assets=n_factors
    )

    factor_benchmark = X.T @ benchmark

    # X.T @ (h - b) <= upper
    # X.T @ h <= upper + X.T @ b
    G_upper = X.T
    u_upper = upper_vec + factor_benchmark

    # X.T @ (h - b) >= lower
    # -X.T @ h <= -lower - X.T @ b
    G_lower = -X.T
    u_lower = -lower_vec - factor_benchmark

    G = np.vstack([G_upper, G_lower])
    u = np.concatenate([u_upper, u_lower])

    if factor_names is None:
        factor_names = [f"factor_{j}" for j in range(n_factors)]

    return LinearConstraints(
        G=G,
        u=u,
        C=np.zeros((0, n_assets)),
        d=np.zeros(0),
        names=[f"{name}_upper" for name in factor_names]
        + [f"{name}_lower" for name in factor_names],
        groups=[group] * (2 * n_factors),
        eq_names=[],
    )


def _expand_bounds(
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    n_assets: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand scalar/vector bounds and validate them."""
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)

    if lower_arr.ndim == 0 and upper_arr.ndim == 0:
        if n_assets is None:
            raise ValueError("n_assets must be provided when bounds are scalars.")
        lower_vec = np.full(n_assets, float(lower_arr))
        upper_vec = np.full(n_assets, float(upper_arr))
    elif lower_arr.ndim == 1 and upper_arr.ndim == 1:
        if lower_arr.shape != upper_arr.shape:
            raise ValueError("lower and upper bounds must have the same shape.")
        lower_vec = lower_arr
        upper_vec = upper_arr
        if n_assets is not None and lower_vec.size != n_assets:
            raise ValueError(
                f"Bounds must have length {n_assets}, got {lower_vec.size}."
            )
    else:
        raise ValueError("lower and upper must both be scalars or both be 1D arrays.")

    if np.any(lower_vec > upper_vec):
        raise ValueError("All lower bounds must be less than or equal to upper bounds.")

    return lower_vec, upper_vec
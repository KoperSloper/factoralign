import numpy as np

from factoralign.alignment import analyze_alignment
from factoralign.constraints import box_bounds, fully_invested
from factoralign.optimizer import solve_mvo
from factoralign.projection import project_onto_factors
from factoralign.risk import RiskModel


def test_risk_model_covariance_and_weights() -> None:
    X = np.array([[1.0], [2.0]])
    factor_cov = np.array([[0.25]])
    specific_var = np.array([0.10, 0.20])

    risk_model = RiskModel(
        X=X,
        factor_cov=factor_cov,
        specific_var=specific_var,
    )

    expected_covariance = X @ factor_cov @ X.T + np.diag(specific_var)

    np.testing.assert_allclose(risk_model.covariance(), expected_covariance)
    np.testing.assert_allclose(
        risk_model.inverse_specific_var_weights(),
        np.array([10.0, 5.0]),
    )


def test_project_onto_factors_returns_orthogonal_residual() -> None:
    vector = np.array([1.0, 2.0, 3.0])
    exposures = np.ones((3, 1))

    result = project_onto_factors(vector, exposures)

    np.testing.assert_allclose(result.coefficients, np.array([2.0]))
    np.testing.assert_allclose(result.spanned, np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(result.orthogonal, np.array([-1.0, 0.0, 1.0]))
    assert result.orthogonality_error(exposures) < 1e-12


def test_constraints_combine_and_slack() -> None:
    constraints = fully_invested(3).combine(box_bounds(0.0, 0.6, n_assets=3))
    holdings = np.array([0.4, 0.3, 0.3])

    assert constraints.n_assets == 3
    assert constraints.n_eq == 1
    assert constraints.n_ineq == 6
    np.testing.assert_allclose(constraints.C @ holdings, constraints.d)
    assert np.all(constraints.slack(holdings) >= 0.0)


def test_solve_mvo_returns_feasible_holdings() -> None:
    alpha = np.array([0.10, 0.02])
    covariance = np.eye(2)
    constraints = fully_invested(2).combine(box_bounds(0.0, 1.0, n_assets=2))

    result = solve_mvo(
        alpha=alpha,
        covariance=covariance,
        constraints=constraints,
        risk_aversion=1.0,
    )

    assert result.status in {"optimal", "optimal_inaccurate"}
    np.testing.assert_allclose(np.sum(result.holdings), 1.0, atol=1e-6)
    assert np.all(result.holdings >= -1e-6)
    assert np.all(result.holdings <= 1.0 + 1e-6)


def test_analyze_alignment_shapes() -> None:
    alpha = np.array([0.05, 0.03, 0.01])
    exposures = np.array(
        [
            [1.0],
            [0.5],
            [0.0],
        ]
    )
    constraints = fully_invested(3).combine(box_bounds(0.0, 0.8, n_assets=3))
    ineq_duals = np.zeros(constraints.n_ineq)
    eq_duals = np.array([0.01])

    analysis = analyze_alignment(
        alpha=alpha,
        constraints=constraints,
        ineq_duals=ineq_duals,
        eq_duals=eq_duals,
        exposures=exposures,
    )

    assert analysis.implied.gamma_ineq.shape == alpha.shape
    assert analysis.implied.gamma_all.shape == alpha.shape
    assert analysis.alpha.orthogonal.shape == alpha.shape
    assert analysis.gamma_ineq.coefficients.shape == (1,)

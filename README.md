# factoralign

`factoralign` is an early-stage Python toolkit for diagnosing and correcting
factor-alignment problems in portfolio optimization.

The project focuses on cases where an alpha signal, constraint pressure, or
optimized portfolio behavior contains components that are not spanned by the
given risk model. It provides small, composable tools for building linear
constraints, solving mean-variance optimization problems, projecting asset-level
vectors onto risk-model factors, and analyzing the orthogonal pieces that remain. It is based on the research of Saxena, Martin, and Stubbs (2013) and Ceria, Saxena, and Stubbs (2012).

## Project Goals

Factor models explain portfolio risk through a set of known exposures. In
practice, useful alpha signals and binding constraints can create exposures to
directions that are not represented in the risk model. `factoralign` helps make
those directions visible.

The package is intended to support workflows such as:

- Constructing canonical portfolio constraints.
- Solving constrained mean-variance optimization problems.
- Decomposing alpha and implied-alpha vectors into factor-spanned and
  factor-orthogonal components.
- Estimating latent factor returns from realized asset returns.
- Augmenting a covariance matrix with an additional aligned factor.

## Main Capabilities

- `factoralign.risk`: `RiskModel` container for linear factor risk models.
- `factoralign.constraints`: linear constraint objects and helpers such as
  `fully_invested`, `box_bounds`, `active_bounds`, and
  `factor_active_bounds`.
- `factoralign.optimizer`: constrained mean-variance optimization with CVXPY.
- `factoralign.projection`: weighted projection of asset-level vectors onto
  the span of risk-model exposures.
- `factoralign.alignment`: diagnostics for alpha, constraint pressure, and
  implied alpha alignment.
- `factoralign.latent`: latent factor return and variance estimation.
- `factoralign.aaf`: covariance augmentation and AAF-based optimization.

## Repository Layout

```text
src/factoralign/        Python package source
notebooks/checks.ipynb  Small validation and smoke-check notebook
notebooks/example.ipynb Worked example notebook
pyproject.toml          Package metadata, dependencies, and tool config
```

## Setup

This package is intended to be installed from the project repository. It is not currently published on PyPI, so installation should be done from source.

### 1. Clone the repository
```bash
git clone https://github.com/KoperSloper/factoralign.git
cd factoralign
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 5. Install the package 

**Standard Installation:**
This installs `factoralign` and its normal runtime dependencies.
```bash
python -m pip install -e .
```

**Development Installation:**
For development tools, tests, and notebooks, install the optional development dependencies instead. This installs `factoralign` plus the dependencies needed for notebooks, testing, linting, and formatting.
```bash
python -m pip install -e ".[dev]"
```

*Note: The package requires Python 3.10 or newer.*

## Dependencies

Core runtime dependencies are defined in `pyproject.toml`:

- NumPy
- pandas
- SciPy
- CVXPY
- scikit-learn

Development dependencies include:

- pytest
- black
- ruff
- matplotlib
- jupyter

## Usage

The example below builds a simple risk model, creates portfolio constraints,
solves a mean-variance optimization problem, and analyzes how the alpha and
constraint pressure align with the risk-model factor space.

```python
import numpy as np

from factoralign.alignment import analyze_alignment, orthogonal_norm_share
from factoralign.constraints import box_bounds, fully_invested
from factoralign.optimizer import solve_mvo
from factoralign.risk import RiskModel

X = np.array(
    [
        [1.0, 0.2],
        [0.8, -0.1],
        [0.4, 0.7],
        [0.1, 1.0],
    ]
)
factor_cov = np.array(
    [
        [0.04, 0.01],
        [0.01, 0.03],
    ]
)
specific_var = np.array([0.05, 0.04, 0.06, 0.05])
alpha = np.array([0.06, 0.04, 0.03, 0.02])

risk_model = RiskModel(
    X=X,
    factor_cov=factor_cov,
    specific_var=specific_var,
)

constraints = fully_invested(n_assets=4).combine(
    box_bounds(lower=0.0, upper=0.5, n_assets=4)
)

result = solve_mvo(
    alpha=alpha,
    covariance=risk_model.covariance(),
    constraints=constraints,
    risk_aversion=5.0,
)

analysis = analyze_alignment(
    alpha=alpha,
    constraints=constraints,
    ineq_duals=result.ineq_duals,
    eq_duals=result.eq_duals,
    exposures=risk_model.X,
    weights=risk_model.inverse_specific_var_weights(),
)

print(result.holdings)
print(orthogonal_norm_share(analysis.alpha))
print(orthogonal_norm_share(analysis.gamma_ineq))
```

See `notebooks/checks.ipynb` and `notebooks/example.ipynb` for more complete
walkthroughs.

## References

Saxena, A., Martin, C., and Stubbs, R. A. (2013). *Constraints in quantitative strategies: An alignment perspective*. Journal of Asset Management, 14, 278–292.

Ceria, S., Saxena, A., and Stubbs, R. A. (2012). *Factor Alignment Problems and Quantitative Portfolio Management*. The Journal of Portfolio Management, 38(2), 29–43.
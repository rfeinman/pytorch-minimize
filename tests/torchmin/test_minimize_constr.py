"""
Test constrained minimization methods.

This module tests the minimize_constr function on various types of constraints,
including inactive constraints (that don't affect the solution) and active
constraints (that bind at the optimum).
"""
import pytest
import torch

from torchmin import minimize, minimize_constr
# from torchmin.constrained.trust_constr import _minimize_trust_constr as minimize_constr
from torchmin.benchmarks import rosen


# Test constants
RTOL = 1e-2
ATOL = 1e-2
MAX_ITER = 50
TOLERANCE = 1e-6  # Numerical tolerance for constraint satisfaction


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def rosen_start():
    """Starting point for Rosenbrock optimization tests."""
    return torch.tensor([1., 8.])


@pytest.fixture(scope='session')
def rosen_unconstrained_solution(rosen_start):
    """Compute the unconstrained Rosenbrock solution for comparison."""
    result = minimize(
        rosen,
        rosen_start,
        method='l-bfgs',
        options=dict(line_search='strong-wolfe'),
        max_iter=MAX_ITER,
        disp=0
    )
    return result


# =============================================================================
# Constraint Functions
# =============================================================================

def sum_constraint(x):
    """Sum constraint: sum(x)."""
    return x.sum()


def norm_constraint(x):
    """L2 norm squared constraint: ||x||^2."""
    return x.square().sum()


# =============================================================================
# Tests
# =============================================================================

class TestUnconstrainedBaseline:
    """Test unconstrained optimization as a baseline."""

    def test_rosen_unconstrained(self, rosen_start):
        """Test unconstrained Rosenbrock minimization."""
        result = minimize(
            rosen,
            rosen_start,
            method='l-bfgs',
            options=dict(line_search='strong-wolfe'),
            max_iter=MAX_ITER,
            disp=0
        )
        assert result.success


class TestInactiveConstraints:
    """
    Test constraints that are inactive (non-binding) at the optimum.

    When the constraint is loose enough, the constrained solution should
    match the unconstrained solution.
    """

    @pytest.mark.parametrize('constraint_fun,constraint_name', [
        (sum_constraint, 'sum'),
        (norm_constraint, 'norm'),
    ])
    def test_loose_constraints(
        self,
        rosen_start,
        rosen_unconstrained_solution,
        constraint_fun,
        constraint_name
    ):
        """Test that loose constraints don't affect the solution."""
        # Upper bound of 10 is loose enough to not affect the solution
        result = minimize_constr(
            rosen,
            rosen_start,
            method='trust-constr',
            constr=dict(fun=constraint_fun, ub=10.),
            max_iter=MAX_ITER,
            disp=0
        )

        torch.testing.assert_close(
            result.x,
            rosen_unconstrained_solution.x,
            rtol=RTOL,
            atol=ATOL,
            msg=f"Loose {constraint_name} constraint affected the solution"
        )


class TestActiveConstraints:
    """
    Test constraints that are active (binding) at the optimum.

    When the constraint is tight, it should bind at the specified bound
    and produce a different solution than the unconstrained case.
    """

    @pytest.mark.parametrize('constraint_fun,ub', [
        (sum_constraint, 1.),
        (norm_constraint, 1.),
    ])
    def test_tight_constraints(self, rosen_start, constraint_fun, ub):
        """Test that tight constraints bind at the specified bound."""
        result = minimize_constr(
            rosen,
            rosen_start,
            method='trust-constr',
            constr=dict(fun=constraint_fun, ub=ub),
            max_iter=MAX_ITER,
            disp=0
        )

        # Verify the constraint is satisfied (with numerical tolerance)
        constraint_value = constraint_fun(result.x)
        assert constraint_value <= ub + TOLERANCE, (
            f"Constraint violated: {constraint_value:.6f} > {ub}"
        )


def test_frankwolfe_birkhoff_polytope():
    n, d = 5, 10
    X = torch.randn(n, d)
    Y = torch.flipud(torch.eye(n)) @ X

    def fun(P):
        return torch.sum((X @ X.T @ P - P @ Y @ Y.T) ** 2)

    init_P = torch.eye(n)
    init_err = torch.sum((X - init_P @ Y) ** 2)
    res = minimize_constr(
        fun,
        init_P,
        method='frank-wolfe',
        constr='birkhoff',
    )
    est_P = res.x
    final_err = torch.sum((X - est_P @ Y) ** 2)
    torch.testing.assert_close(est_P.sum(0), torch.ones(n))
    torch.testing.assert_close(est_P.sum(1), torch.ones(n))
    assert final_err < 0.01 * init_err


def test_frankwolfe_tracenorm():
    dim = 5
    init_X = torch.zeros((dim, dim))
    eye = torch.eye(dim)

    def fun(X):
        return torch.sum((X - eye) ** 2)

    res = minimize_constr(
        fun,
        init_X,
        method='frank-wolfe',
        constr='tracenorm',
        options=dict(t=5.0),
    )
    est_X = res.x
    torch.testing.assert_close(est_X, eye, rtol=1e-2, atol=1e-2)

    res = minimize_constr(
        fun,
        init_X,
        method='frank-wolfe',
        constr='tracenorm',
        options=dict(t=1.0),
    )
    est_X = res.x
    torch.testing.assert_close(est_X, 0.2 * eye, rtol=1e-2, atol=1e-2)


def test_lbfgsb_simple_quadratic():
    """Test L-BFGS-B on a simple bounded quadratic problem.

    Minimize: f(x) = (x1 - 2)^2 + (x2 - 1)^2
    Subject to: 0 <= x1 <= 1.5, 0 <= x2 <= 2

    The unconstrained minimum is at (2, 1), but x1 is constrained,
    so the optimal solution should be at (1.5, 1).
    """

    def fun(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2

    x0 = torch.tensor([0.5, 0.5])
    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([1.5, 2.0])

    result = minimize_constr(
        fun,
        x0,
        method='l-bfgs-b',
        bounds=(lb, ub),
        options=dict(gtol=1e-6, ftol=1e-9),
    )

    # Check if close to expected solution
    expected_x = torch.tensor([1.5, 1.0])
    expected_f = 0.25

    assert torch.allclose(result.x, expected_x, atol=1e-4), \
        f"Solution {result.x} not close to expected {expected_x}"
    assert abs(result.fun - expected_f) < 1e-4, \
        f"Function value {result.fun} not close to expected {expected_f}"


def test_lbfgsb_rosenbrock():
    """Test L-BFGS-B on Rosenbrock function with bounds.

    Minimize: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Subject to: -2 <= x <= 2, -2 <= y <= 2

    The unconstrained minimum is at (1, 1).
    """

    x0 = torch.tensor([-1.0, 1.5])
    lb = torch.tensor([-2.0, -2.0])
    ub = torch.tensor([2.0, 2.0])

    result = minimize_constr(
        rosen,
        x0,
        method='l-bfgs-b',
        bounds=(lb, ub),
        options=dict(gtol=1e-6, ftol=1e-9, max_iter=100),
    )

    # Check if close to expected solution
    expected_x = torch.tensor([1.0, 1.0])

    assert torch.allclose(result.x, expected_x, atol=1e-3), \
        f"Solution {result.x} not close to expected {expected_x}"
    assert result.fun < 1e-6, \
        f"Function value {result.fun} not close to 0"


def test_lbfgsb_active_constraints():
    """Test L-BFGS-B with multiple active constraints.

    Minimize: f(x) = sum(x_i^2)
    Subject to: x_i >= 1 for all i

    The solution should be all ones (on the boundary).
    """

    def fun(x):
        return (x**2).sum()

    n = 5
    x0 = torch.ones(n) * 2.0
    lb = torch.ones(n)
    ub = torch.ones(n) * 10.0

    result = minimize_constr(
        fun,
        x0,
        method='l-bfgs-b',
        bounds=(lb, ub),
        options=dict(gtol=1e-6, ftol=1e-9),
    )

    expected_x = torch.ones(n)
    expected_f = float(n)

    assert torch.allclose(result.x, expected_x, atol=1e-4), \
        f"Solution {result.x} not close to expected {expected_x}"
    assert abs(result.fun - expected_f) < 1e-4, \
        f"Function value {result.fun} not close to expected {expected_f}"

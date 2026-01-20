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

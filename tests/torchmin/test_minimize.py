"""
Test unconstrained minimization methods on various objective functions.

This module tests all unconstrained optimization methods provided by torchmin
on a variety of test problems. To add a new test problem:
1. Create a fixture in conftest.py (or here) following the standard format
2. Add the fixture name to the PROBLEMS list below

NOTE: The problem fixtures are defined in `conftest.py`
"""
import pytest
import torch

from torchmin import minimize


# All unconstrained optimization methods
ALL_METHODS = [
    'bfgs',
    'l-bfgs',
    'cg',
    'newton-cg',
    'newton-exact',
    'trust-ncg',
    # 'trust-krylov',  # TODO: fix trust-krylov solver and add this back
    'trust-exact',
    'dogleg',
]

# All test problems - add new problem fixture names here
PROBLEMS = [
    'least_squares_problem',
    'rosenbrock_problem',
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def problem(request):
    """
    Indirect fixture that routes to specific problem fixtures.

    This allows parametrizing over multiple problem fixtures without
    duplicating test code.
    """
    return request.getfixturevalue(request.param)


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.parametrize('problem', PROBLEMS, indirect=True)
@pytest.mark.parametrize('method', ALL_METHODS)
def test_minimize(method, problem):
    """Test minimization methods on various optimization problems."""
    result = minimize(problem['objective'], problem['x0'], method=method)

    # TODO: should we check result.success??
    # assert result.success, (
    #     f"Optimization failed for method {method} on {problem['name']}: "
    #     f"{result.message}"
    # )

    torch.testing.assert_close(
        result.x, problem['solution'],
        rtol=1e-4, atol=1e-3,
        msg=f"Solution incorrect for method {method} on {problem['name']}"
    )

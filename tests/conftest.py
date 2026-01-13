"""Shared pytest fixtures for torchmin tests."""
import pytest
import torch

from torchmin.benchmarks import rosen


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    yield 42


# =============================================================================
# Objective Function Fixtures
# =============================================================================
# To add a new test problem, create a fixture that returns a dict with:
#   - 'objective': callable, the objective function
#   - 'x0': Tensor, initial point
#   - 'solution': Tensor, known optimal solution
#   - 'name': str, descriptive name for the problem


@pytest.fixture(scope='session')
def least_squares_problem():
    """
    Generate a least squares problem for testing optimization algorithms.

    Creates a linear regression problem: min ||Y - X @ B||^2
    where X is N x D, Y is N x M, and B is D x M.

    This is a session-scoped fixture, so the same problem instance is used
    across all tests for consistency.

    Returns
    -------
    dict
        Dictionary containing:
        - objective: callable, the objective function
        - x0: Tensor, initial parameter values (zeros)
        - solution: Tensor, the true solution
        - X: Tensor, design matrix
        - Y: Tensor, target values
    """
    torch.manual_seed(42)
    N, D, M = 100, 7, 5
    X = torch.randn(N, D)
    Y = torch.randn(N, M)

    def objective(B):
        return torch.sum((Y - X @ B) ** 2)

    # target B
    #trueB = torch.linalg.inv(X.T @ X) @ X.T @ Y
    trueB = torch.linalg.lstsq(X, Y).solution # XB = Y (solve for B)

    # initial B
    B0 = torch.zeros(D, M)

    return {
        'objective': objective,
        'x0': B0,
        'solution': trueB,
        'X': X,
        'Y': Y,
        'name': 'least_squares',
    }


@pytest.fixture(scope='session')
def rosenbrock_problem():
    """Rosenbrock function (banana function)."""

    torch.manual_seed(42)
    D = 10
    x0 = torch.zeros(D)
    x_sol = torch.ones(D)

    return {
        'objective': rosen,
        'x0': x0,
        'solution': x_sol,
        'name': 'rosenbrock',
    }


# =============================================================================
# Other Fixtures
# =============================================================================


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    """
    Parametrize tests across CPU and CUDA devices.

    Automatically skips CUDA tests if CUDA is not available.
    """
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return torch.device(request.param)

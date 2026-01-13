"""Test that all public APIs are importable and accessible."""
import pytest


def test_import_main_package():
    """Test importing the main torchmin package."""
    import torchmin
    assert hasattr(torchmin, '__version__')


def test_import_core_functions():
    """Test importing core minimize functions."""
    from torchmin import minimize, minimize_constr, Minimizer


def test_import_benchmarks():
    """Test importing benchmark functions."""
    from torchmin.benchmarks import rosen


@pytest.mark.parametrize('method', [
    'bfgs',
    'l-bfgs',
    'cg',
    'newton-cg',
    'newton-exact',
    'trust-ncg',
    # 'trust-krylov',
    'trust-exact',
    'dogleg',
])
def test_method_available(method):
    """Test that all advertised methods are available and callable."""
    import torch
    from torchmin import minimize

    # Simple quadratic objective: f(x) = ||x||^2
    x0 = torch.zeros(2)
    result = minimize(lambda x: x.square().sum(), x0, method=method, max_iter=1)
    assert result is not None

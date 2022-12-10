import pytest
import torch

from torchmin import minimize, minimize_constr
from torchmin.benchmarks import rosen


def test_rosen():
    """Test Rosenbrock problem with constraints."""

    x0 = torch.tensor([1., 8.])
    res = minimize(
        rosen, x0,
        method='l-bfgs',
        options=dict(line_search='strong-wolfe'),
        max_iter=50,
        disp=0
    )


    # Test inactive constraints

    res_constrained_sum = minimize_constr(
        rosen, x0,
        constr=dict(fun=lambda x: x.sum(), ub=10.),
        max_iter=50,
        disp=0
    )
    torch.testing.assert_close(
        res.x, res_constrained_sum.x, rtol=1e-2, atol=1e-2)

    res_constrained_norm = minimize_constr(
        rosen, x0,
        constr=dict(fun=lambda x: x.square().sum(), ub=10.),
        max_iter=50,
        disp=0
    )
    torch.testing.assert_close(
        res.x, res_constrained_norm.x, rtol=1e-2, atol=1e-2)


    # Test active constraints

    res_constrained_sum = minimize_constr(
        rosen, x0,
        constr=dict(fun=lambda x: x.sum(), ub=1.),
        max_iter=50,
        disp=0
    )
    assert res_constrained_sum.x.sum() <= 1.
    res_constrained_norm = minimize_constr(
        rosen, x0,
        constr=dict(fun=lambda x: x.square().sum(), ub=1.),
        max_iter=50,
        disp=0
    )
    assert res_constrained_norm.x.square().sum() <= 1.

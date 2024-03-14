import pytest
import torch

from torchmin import (
    minimize_constr_birkhoff_polytope,
    minimize_constr_tracenorm,
)


def test_birkhoff_polytope():
    n, d = 5, 10
    X = torch.randn(n, d)
    Y = torch.flipud(torch.eye(n)) @ X
    def fun(P):
        return torch.sum((X @ X.T @ P - P @ Y @ Y.T) ** 2)

    init_P = torch.eye(n)
    init_err = torch.sum((X - init_P @ Y) ** 2)
    res = minimize_constr_birkhoff_polytope(fun, init_P)
    est_P = res.x
    final_err = torch.sum((X - est_P @ Y) ** 2)
    torch.testing.assert_close(est_P.sum(0), torch.ones(n))
    torch.testing.assert_close(est_P.sum(1), torch.ones(n))
    assert final_err < 0.01 * init_err


def test_tracenorm():
    def fun(X):
        return torch.sum((X - torch.eye(5)) ** 2)

    init_X = torch.zeros((5, 5))
    res = minimize_constr_tracenorm(fun, init_X, 5.)
    est_X = res.x
    torch.testing.assert_close(est_X, torch.eye(5), rtol=1e-2, atol=1e-2)

    res = minimize_constr_tracenorm(fun, init_X, 1.)
    est_X = res.x
    torch.testing.assert_close(est_X, 0.2*torch.eye(5), rtol=1e-2, atol=1e-2)

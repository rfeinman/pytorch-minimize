import warnings
import torch
from torch.optim.lbfgs import _strong_wolfe
from scipy.optimize import minimize_scalar


def strong_wolfe(fun, x, t, d, f, g, gtd, **kwargs):
    """
    Expects `fun` to take arguments {x, t, d} and return {f(x1), f'(x1)},
    where x1 is the new location after taking a step from x in direction d
    with step size t.
    """
    return _strong_wolfe(fun, x, t, d, f, g, gtd, **kwargs)


def brent(fun, x, d, bounds=(0,10)):
    """
    Expects `fun` to take arguments {x} and return {f(x)}
    """
    def line_obj(t):
        return float(fun(x + t * d))
    res = minimize_scalar(line_obj, bounds=bounds, method='bounded')
    return res.x


def backtracking(fun, x, t, d, f, g, mu=0.1, decay=0.98, max_ls=500, tmin=1e-5):
    """
    Expects `fun` to take arguments {x, t, d} and return {f(x1), x1},
    where x1 is the new location after taking a step from x in direction d
    with step size t.
    """
    x_new = x
    f_new = f
    gtd = g.mul(d).sum()
    success = True
    for i in range(max_ls):
        f_new, x_new = fun(x, t, d)
        if f_new <= f + mu * t * gtd:
            break
        if t <= tmin:
            warnings.warn('step size has reached the minimum threshold.')
            success = False
            break
        t = t.mul(decay)
    else:
        warnings.warn('backtracking did not converge.')
        success = False

    return x_new, f_new, t, success
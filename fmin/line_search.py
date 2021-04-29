import warnings
import torch
from torch.optim.lbfgs import _strong_wolfe, _cubic_interpolate
from scipy.optimize import minimize_scalar

__all__ = ['strong_wolfe', 'brent', 'backtracking']


def _strong_wolfe_extra(
        obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9,
        tolerance_change=1e-9, max_ls=25, extra_condition=None):
    """A modified variant of pytorch's strong-wolfe line search that supports
    an "extra_condition" argument (callable).

    This is required for methods such as Conjugate Gradient (polak-ribiere)
    where the strong wolfe conditions do not guarantee that we have a
    descent direction.

    Code borrowed from pytorch::
        Copyright (c) 2016 Facebook, Inc.
        All rights reserved.
    """
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    if extra_condition is None:
        extra_condition = lambda *args: True
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


def strong_wolfe(fun, x, t, d, f, g, gtd=None, **kwargs):
    """
    Expects `fun` to take arguments {x, t, d} and return {f(x1), f'(x1)},
    where x1 is the new location after taking a step from x in direction d
    with step size t.
    """
    if gtd is None:
        gtd = g.mul(d).sum()

    # use python floats for scalars as per torch.optim.lbfgs
    f, t = float(f), float(t)

    if 'extra_condition' in kwargs:
        f, g, t, ls_nevals = _strong_wolfe_extra(
            fun, x.view(-1), t, d.view(-1), f, g.view(-1), gtd, **kwargs)
    else:
        # in theory we shouldn't need to use pytorch's native _strong_wolfe
        # since the custom implementation above is equivalent with
        # extra_codition=None. But we will keep this in case they make any
        # changes.
        f, g, t, ls_nevals = _strong_wolfe(
            fun, x.view(-1), t, d.view(-1), f, g.view(-1), gtd, **kwargs)

    # convert back to torch scalar
    f = torch.as_tensor(f, dtype=x.dtype, device=x.device)

    return f, g.view_as(x), t, ls_nevals


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

    We use a generalized variant of the armijo condition that supports
    arbitrary step functions x' = step(x,t,d). When step(x,t,d) = x + t * d
    it is equivalent to the standard condition.
    """
    x_new = x
    f_new = f
    success = False
    for i in range(max_ls):
        f_new, x_new = fun(x, t, d)
        if f_new <= f + mu * g.mul(x_new-x).sum():
            success = True
            break
        if t <= tmin:
            warnings.warn('step size has reached the minimum threshold.')
            break
        t = t.mul(decay)
    else:
        warnings.warn('backtracking did not converge.')

    return x_new, f_new, t, success
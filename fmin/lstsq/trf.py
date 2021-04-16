"""Trust Region Reflective algorithm for least-squares optimization.
"""
import torch
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._lsq.common import (print_header_nonlinear,
                                        print_iteration_nonlinear)

from .lsmr import lsmr
from .linear_operator import jacobian_linop, jacobian_dense
from .common import (right_multiplied_operator, build_quadratic_1d,
                     minimize_quadratic_1d, evaluate_quadratic,
                     solve_trust_region_2d, check_termination,
                     update_tr_radius, solve_lsq_trust_region)


def trf(fun, x0, f0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
        tr_solver, tr_options, verbose):
    # For efficiency, it makes sense to run the simplified version of the
    # algorithm when no bounds are imposed. We decided to write the two
    # separate functions. It violates the DRY principle, but the individual
    # functions are kept the most readable.
    if lb.isneginf().all() and ub.isposinf().all():
        return trf_no_bounds(
            fun, x0, f0, ftol, xtol, gtol, max_nfev, x_scale,
            tr_solver, tr_options, verbose)
    else:
        raise NotImplementedError('trf with bounds not currently supported.')


def trf_no_bounds(fun, x0, f0=None, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                  max_nfev=None, x_scale=1.0, tr_solver='lsmr',
                  tr_options=None, verbose=0):
    if max_nfev is None:
        max_nfev = x0.numel() * 100
    if tr_options is None:
        tr_options = {}
    assert tr_solver in ['exact', 'lsmr']
    if tr_solver == 'exact':
        jacobian = jacobian_dense
    else:
        jacobian = jacobian_linop

    x = x0.clone()
    if f0 is None:
        f = fun(x)
    else:
        f = f0
    f_true = f.clone()
    J = jacobian(fun, x)
    nfev = njev = 1
    m, n = J.shape

    cost = 0.5 * f.dot(f)
    g = J.T.mv(f)

    scale, scale_inv = x_scale, 1 / x_scale
    Delta = (x0 * scale_inv).norm()
    if Delta == 0:
        Delta = 1.0

    if tr_solver == 'lsmr':
        damp = tr_options.pop('damp', 0.0)
        regularize = tr_options.pop('regularize', True)
        reg_term = 0.

    alpha = 0.  # "Levenberg-Marquardt" parameter
    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        g_norm = g.norm(np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                      step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        d = scale
        g_h = d * g

        if tr_solver == 'exact':
            J_h = J * d
            U, s, V = torch.linalg.svd(J_h, full_matrices=False)
            V = V.T
            uf = U.T.mv(f)
        elif tr_solver == 'lsmr':
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h)
                to_tr = Delta / g_h.norm()
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta**2

            damp_full = (damp**2 + reg_term)**0.5
            gn_h = lsmr(J_h, f, damp=damp_full, **tr_options)[0]
            S = torch.vstack((g_h, gn_h)).T  # [dim, 2]
            S, _ = torch.linalg.qr(S, mode='reduced')
            JS = J_h.matmul(S)  # TODO: can we avoid jacobian mm?
            B_S = JS.T.matmul(JS)
            g_S = S.T.matmul(g_h)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == 'exact':
                step_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == 'lsmr':
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                step_h = S.matmul(p_S)

            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = step_h.norm()

            if not f_new.isfinite().all():
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            cost_new = 0.5 * f_new.dot(f_new)
            actual_reduction = cost - cost_new

            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = step.norm()
            termination_status = check_termination(
                actual_reduction, cost, step_norm, x.norm(), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new
            f = f_new
            f_true = f.clone()
            cost = cost_new
            J = jacobian(fun, x)
            g = J.T.mv(f)
            njev += 1
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = torch.zeros_like(x)
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status)

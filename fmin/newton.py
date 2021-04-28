from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
from torch import Tensor
import torch

from .function import ScalarFunction, DirectionalEvaluate
from .line_search import strong_wolfe

_status_message['cg_warn'] = "Warning: CG iterations didn't converge. The " \
                             "Hessian is not positive definite."


def _cg_iters(grad, hess, max_iter, normp=1):
    """A CG solver specialized for the NewtonCG sub-problem.

    Derived from Algorithm 7.1 of "Numerical Optimization (2nd Ed.)"
    (Nocedal & Wright, 2006; pp. 169)
    """
    # generalized dot product that supports batch inputs
    # TODO: let the user specify dot fn?
    dot = lambda u,v: u.mul(v).sum(-1, keepdim=True)

    g_norm = grad.norm(p=normp)
    tol = g_norm * g_norm.sqrt().clamp(0, 0.5)
    eps = torch.finfo(grad.dtype).eps
    n_iter = 0  # TODO: remove?
    maxiter_reached = False

    # initialize state and iterate
    x = torch.zeros_like(grad)
    r = grad.clone()
    p = grad.neg()
    rs = dot(r, r)
    for n_iter in range(max_iter):
        if r.norm(p=normp) < tol:
            break
        Bp = hess.mv(p)
        curv = dot(p, Bp)
        curv_sum = curv.sum()
        if curv_sum < 0:
            # hessian is not positive-definite
            if n_iter == 0:
                # if first step, fall back to steepest descent direction
                # (scaled by Rayleigh quotient)
                x = grad.mul(rs / curv)
                #x = grad.neg()
            break
        elif curv_sum <= 3 * eps:
            break
        alpha = rs / curv
        x.addcmul_(alpha, p)
        r.addcmul_(alpha, Bp)
        rs_new = dot(r, r)
        p.mul_(rs_new / rs).sub_(r)
        rs = rs_new
    else:
        # curvature keeps increasing; bail
        maxiter_reached = True

    return x, n_iter, maxiter_reached


@torch.no_grad()
def _minimize_newton_cg(
        f, x0, lr=1., max_iter=None, cg_max_iter=None,
        twice_diffable=True, line_search='strong-wolfe', xtol=1e-5,
        normp=1, callback=None, disp=0, return_all=False):
    """
    Minimize a scalar function of one or more variables using the
    Newton-Raphson method, with Conjugate Gradient for the linear inverse
    sub-problem.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * x0.numel()
    cg_max_iter : int, optional
        Maximum number of iterations for CG subproblem. Recommended to
        leave this at the default of 20 * x0.numel()
    twice_diffable : bool
        Whether to assume the function is twice continuously differentiable.
        If True, hessian-vector products will be much faster.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    normp : int | float | str
        The norm type to use for termination conditions. Can be any value
        supported by `torch.norm` p argument.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. callback(x_k)
    disp : int | bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)
    xtol = x0.numel() * xtol
    if max_iter is None:
        max_iter = x0.numel() * 200
    if cg_max_iter is None:
        cg_max_iter = x0.numel() * 20

    # construct scalar objective function
    f_closure = ScalarFunction(f, hessp=True, twice_diffable=twice_diffable)
    if line_search == 'strong-wolfe':
        dir_evaluate = DirectionalEvaluate(f, x_shape=x0.shape)

    # initial settings
    x = x0.detach().clone(memory_format=torch.contiguous_format)
    fval, grad, hessp, _ = f_closure(x)
    if return_all:
        allvecs = [x]
    nfev = 1  # number of function evals
    ncg = 0   # number of cg iterations
    n_iter = 0

    if disp > 1:
        print('initial fval: %0.4f' % fval)

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % n_iter)
            print("         Function evaluations: %d" % nfev)
            print("         CG iterations: %d" % ncg)
        result = OptimizeResult(fun=fval, jac=grad, nfev=nfev, ncg=ncg,
                                status=warnflag, success=(warnflag==0),
                                message=msg, x=x, nit=n_iter)
        if return_all:
            result['allvecs'] = allvecs
        return result


    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ============================================================
        #  Compute a search direction pk by applying the CG method to
        #       H_f(xk) p = - J_f(xk) starting from 0.
        # ============================================================

        # Compute search direction with conjugate gradient (GG)
        d, cg_iters, cg_fail = _cg_iters(grad, hessp, cg_max_iter, normp)
        ncg += cg_iters

        if cg_fail:
            return terminate(3, _status_message['cg_warn'])


        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == 'none':
            update = d.mul(lr)
            x = x + update
        elif line_search == 'strong-wolfe':
            # strong-wolfe line search
            _, _, t, ls_nevals = \
                strong_wolfe(dir_evaluate, x, lr, d, fval, grad)
            nfev += ls_nevals
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        # re-evaluate function
        fval, grad, hessp, _ = f_closure(x)
        nfev += 1

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (n_iter, fval))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)


        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=normp) <= xtol:
            return terminate(0, _status_message['success'])

        if not fval.isfinite():
            return terminate(3, _status_message['nan'])

    # if we get to the end, the maximum num. iterations was reached
    return terminate(1, "Warning: " + _status_message['maxiter'])



@torch.no_grad()
def _minimize_newton_exact(
        f, x0, lr=1., max_iter=None, line_search='strong-wolfe', xtol=1e-5,
        normp=1, tikhonov=0., handle_npd='grad', callback=None, disp=0,
        return_all=False):
    """
    Minimize a scalar function of one or more variables using the
    Newton-Raphson method.

    This variant uses an "exact" Newton routine based on Cholesky factorization
    of the explicit Hessian matrix. In general it will be less efficient than
    NewtonCG, and less robust to non-PD Hessians.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * x0.numel()
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    normp : int | float | str
        The norm type to use for termination conditions. Can be any value
        supported by `torch.norm` p argument.
    tikhonov : float
        Optional diagonal regularization (Tikhonov) parameter for the Hessian.
    handle_npd : str
        Mode for handling non-positive definite hessian matrices. Can be one
        of the following:
            'grad' : use steepest descent direction (gradient)
            'lu' : solve the inverse hessian with LU factorization
            'eig' : use symmetric eigendecomposition to determine a
                    diagonal regularization parameter
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. callback(x_k)
    disp : int | bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)
    xtol = x0.numel() * xtol
    if max_iter is None:
        max_iter = x0.numel() * 200

    # Construct scalar objective function
    f_closure = ScalarFunction(f, x_shape=x0.shape, hess=True)
    if line_search == 'strong-wolfe':
        dir_evaluate = DirectionalEvaluate(f, x_shape=x0.shape)

    # initial settings
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    fval, grad, _, hess = f_closure(x)
    if tikhonov > 0:
        hess.diagonal().add_(tikhonov)
    if disp > 1:
        print('initial fval: %0.4f' % fval)
    if return_all:
        allvecs = [x]
    nfev = 1  # number of function evals
    nfail = 0
    n_iter = 0


    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % n_iter)
            print("         Function evaluations: %d" % nfev)
        result = OptimizeResult(fun=fval, jac=grad, nfev=nfev, nfail=nfail,
                                status=warnflag, success=(warnflag==0),
                                message=msg, x=x.view_as(x0), nit=n_iter)
        if return_all:
            result['allvecs'] = allvecs
        return result


    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ==================================================
        #  Compute a search direction d by solving
        #          H_f(x) d = - J_f(x)
        #  with the true Hessian and Cholesky factorization
        # ===================================================

        # Compute search direction with Cholesky solve
        try:
            d = torch.cholesky_solve(grad.neg().unsqueeze(1),
                                     torch.linalg.cholesky(hess)).squeeze(1)
            chol_fail = False
        except:
            chol_fail = True
            nfail += 1
            if handle_npd == 'lu':
                d = torch.linalg.solve(hess, grad.neg())
            elif handle_npd == 'grad':
                d = grad.neg()
            elif handle_npd == 'eig':
                # this setting is experimental! use with caution
                eig, V = torch.linalg.eigh(hess)
                tau = torch.clamp(-1.5 * eig[0], min=1e-3)
                eig.add_(tau)
                #grad.add_(x, alpha=tau)
                d = - V.mv(V.t().mv(grad) / eig)
            else:
                raise RuntimeError('invalid handle_npd encountered.')


        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == 'none':
            update = d.mul(lr)
            x = x + update
        elif line_search == 'strong-wolfe':
            # strong-wolfe line search
            _, _, t, ls_nevals = \
                strong_wolfe(dir_evaluate, x, lr, d, fval, grad)
            nfev += ls_nevals
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        # ===================================
        #  Re-evaluate func/Jacobian/Hessian
        # ===================================

        fval, grad, _, hess = f_closure(x)
        if tikhonov > 0:
            hess.diagonal().add_(tikhonov)
        nfev += 1

        if disp > 1:
            print('iter %3d - fval: %0.4f - chol_fail: %r' % (n_iter, fval, chol_fail))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)

        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=normp) <= xtol:
            return terminate(0, _status_message['success'])

        if not fval.isfinite():
            return terminate(3, _status_message['nan'])

    # if we get to the end, the maximum num. iterations was reached
    return terminate(1, "Warning: " + _status_message['maxiter'])
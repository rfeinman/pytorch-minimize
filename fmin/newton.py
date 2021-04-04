import warnings
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
from torch import Tensor
import torch
import torch.autograd as autograd
from torch._vmap_internals import _vmap

from .line_search import strong_wolfe

_status_message['cg_warn'] = "Warning: CG iterations didn't converge. The " \
                             "Hessian is not positive definite."


def _cg_iters(grad, hvp, max_iter, norm_p=1):
    """A CG solver specialized for the NewtonCG sub-problem.

    Derived from Algorithm 7.1 of "Numerical Optimization (2nd Ed.)"
    (Nocedal & Wright, 2006; pp. 169)
    """
    # generalized dot product that supports batch inputs
    # TODO: let the user specify dot fn?
    dot = lambda u,v: u.mul(v).sum(-1, keepdim=True)

    g_norm = grad.norm(p=norm_p)
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
        if r.norm(p=norm_p) < tol:
            break
        Bp = hvp(p)
        curv = dot(p, Bp)
        curv_sum = curv.sum()
        if curv_sum < 0:
            if n_iter == 0:
                # hessian is not positive-definite; fall back to steepest
                # descent direction (scaled by Rayleigh quotient)
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
def fmin_newton_cg(
        f, x0, lr=1., max_iter=None, cg_options=None,
        twice_diffable=True, line_search='strong-wolfe', xtol=1e-5,
        callback=None, disp=0, return_all=False):
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
    cg_options : dict, optional
        A dictionary of keyword arguments to pass to the CG solver. Available
        arguments are {'max_iter', 'rtol', 'tol'}. In general it is
        recommended to leave these at their default values.
    twice_diffable : bool
        Whether to assume the function is twice continuously differentiable.
        If True, hessian-vector products will be much faster.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
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
    if cg_options is None:
        cg_options = {}
    cg_options.setdefault('max_iter', x0.numel() * 20)
    cg_options.setdefault('norm_p', 1)

    def f_with_grad(x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        grad, = autograd.grad(fval, x)
        return fval.detach(), grad

    def dir_evaluate(x, t, d):
        """directional evaluate. Used only for strong-wolfe line search"""
        x = x.add(d, alpha=t).view_as(x0)
        fval, grad = f_with_grad(x)
        return fval, grad.view(-1)

    # initial settings
    x = x0.detach().clone(memory_format=torch.contiguous_format)
    if return_all:
        allvecs = [x]
    fval = x.new_tensor(-1.)
    grad = x.new_full(x.shape, -1.)
    nfev = 0  # number of function evals
    ncg = 0   # number of cg iterations
    n_iter = 0

    if disp > 1:
        fval = f(x)
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

        # Compute f(xk) and f'(xk)
        x.requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
            # Note: create graph of gradient to enable hessian-vector product
            grad, = autograd.grad(fval, x, create_graph=True)
        nfev += 1

        # Initialize hessian-vector product function.
        # Note: PyTorch `hvp` is significantly slower than `vhp` due to
        # backward mode AD constraints. If the function is twice continuously
        # differentiable, then hvp = vhp^T, and we can use the faster vhp.
        if twice_diffable:
            hvp = lambda v: autograd.grad(grad, x, v, retain_graph=True)[0]
        else:
            g_grad = torch.zeros_like(grad, requires_grad=True)
            with torch.enable_grad():
                g_x = autograd.grad(grad, x, g_grad, create_graph=True)[0]
            hvp = lambda v: autograd.grad(g_x, g_grad, v, retain_graph=True)[0]

        # Compute search direction with conjugate gradient (GG)
        d, cg_iters, cg_fail = _cg_iters(grad.detach(), hvp, **cg_options)
        ncg += cg_iters
        if cg_fail:
            return terminate(3, _status_message['cg_warn'])

        # Free the autograd graph
        x = x.detach()
        fval = fval.detach()
        grad = grad.detach()


        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == 'none':
            update = d.mul(lr)
            x = x + update
            fval = f(x)
        elif line_search == 'strong-wolfe':
            # strong-wolfe line search
            gtd = grad.mul(d).sum()
            fval, grad, t, ls_nevals = \
                strong_wolfe(dir_evaluate, x.view(-1), lr, d.view(-1), fval,
                             grad.view(-1), gtd)
            grad = grad.view_as(x)
            nfev += ls_nevals
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (n_iter, fval))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)


        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=1) <= xtol:
            return terminate(0, _status_message['success'])

        if not fval.isfinite():
            return terminate(3, _status_message['nan'])

    # if we get to the end, the maximum num. iterations was reached
    return terminate(1, "Warning: " + _status_message['maxiter'])



@torch.no_grad()
def fmin_newton_exact(
        f, x0, lr=1., max_iter=None, line_search='strong-wolfe', xtol=1e-5,
        tikhonov=0., callback=None, disp=0, return_all=False):
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
    tikhonov : float
        Optional diagonal regularization (Tikhonov) parameter for the Hessian.
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
    # identity matrix buffer to use for hessian computation
    I = torch.eye(x0.numel(), dtype=x0.dtype, device=x0.device)

    def dir_evaluate(x, t, d):
        """directional evaluate. Used only for strong-wolfe line search"""
        x = x.add(d, alpha=t)
        x = x.view_as(x0).detach().requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        grad, = autograd.grad(fval, x)
        return fval.detach(), grad.view(-1)

    def f_with_hess(x):
        x = x.view_as(x0).detach().requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
            grad = autograd.grad(fval, x, create_graph=True)[0].view(-1)
            hess = _vmap(lambda v: autograd.grad(grad, x, v)[0])(I)
        hess = hess.view(x0.numel(), x0.numel())
        if tikhonov > 0:
            hess.diagonal().add_(tikhonov)

        return fval.detach(), grad.detach(), hess

    # initial settings
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    fval, grad, hess = f_with_hess(x)
    if disp > 1:
        print('initial fval: %0.4f' % fval)
    if return_all:
        allvecs = [x]
    nfev = 1  # number of function evals
    n_iter = 0


    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % n_iter)
            print("         Function evaluations: %d" % nfev)
        result = OptimizeResult(fun=fval, jac=grad, nfev=nfev,
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
        d = grad.neg().unsqueeze(1)
        try:
            d = torch.cholesky_solve(d, torch.linalg.cholesky(hess))
        except:
            warnings.warn('Cholesky factorization failed. Resorting to '
                          'LU factorization...')
            d = torch.linalg.solve(hess, d)
        d = d.squeeze(1)


        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == 'none':
            update = d.mul(lr)
            x = x + update
        elif line_search == 'strong-wolfe':
            # strong-wolfe line search
            gtd = grad.mul(d).sum()
            _, _, t, ls_nevals = \
                strong_wolfe(dir_evaluate, x, lr, d, fval, grad, gtd)
            nfev += ls_nevals
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        # ===================================
        #  Re-evaluate func/Jacobian/Hessian
        # ===================================

        fval, grad, hess = f_with_hess(x)
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

        if update.norm(p=1) <= xtol:
            return terminate(0, _status_message['success'])

        if not fval.isfinite():
            return terminate(3, _status_message['nan'])

    # if we get to the end, the maximum num. iterations was reached
    return terminate(1, "Warning: " + _status_message['maxiter'])
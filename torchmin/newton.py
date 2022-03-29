from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import eigsh
from torch import Tensor
import torch

from .function import ScalarFunction
from .line_search import strong_wolfe

try:
    from scipy.optimize.optimize import _status_message
except ImportError:
    from scipy.optimize._optimize import _status_message

_status_message['cg_warn'] = "Warning: CG iterations didn't converge. The " \
                             "Hessian is not positive definite."


def _cg_iters(grad, hess, max_iter, normp=1):
    """A CG solver specialized for the NewtonCG sub-problem.

    Derived from Algorithm 7.1 of "Numerical Optimization (2nd Ed.)"
    (Nocedal & Wright, 2006; pp. 169)
    """
    # Get the most efficient dot product method for this problem
    if grad.dim() == 1:
        # standard dot product
        dot = torch.dot
    elif grad.dim() == 2:
        # batched dot product
        dot = lambda u,v: torch.bmm(u.unsqueeze(1), v.unsqueeze(2)).view(-1,1)
    else:
        # generalized dot product that supports batch inputs
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
        fun, x0, lr=1., max_iter=None, cg_max_iter=None,
        twice_diffable=True, line_search='strong-wolfe', xtol=1e-5,
        normp=1, callback=None, disp=0, return_all=False):
    """Minimize a scalar function of one or more variables using the
    Newton-Raphson method, with Conjugate Gradient for the linear inverse
    sub-problem.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    cg_max_iter : int, optional
        Maximum number of iterations for CG subproblem. Recommended to
        leave this at the default of ``20 * x0.numel()``.
    twice_diffable : bool
        Whether to assume the function is twice continuously differentiable.
        If True, hessian-vector products will be much faster.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
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
    sf = ScalarFunction(fun, x0.shape, hessp=True, twice_diffable=twice_diffable)
    closure = sf.closure
    if line_search == 'strong-wolfe':
        dir_evaluate = sf.dir_evaluate

    # initial settings
    x = x0.detach().clone(memory_format=torch.contiguous_format)
    f, g, hessp, _ = closure(x)
    if disp > 1:
        print('initial fval: %0.4f' % f)
    if return_all:
        allvecs = [x]
    ncg = 0   # number of cg iterations
    n_iter = 0

    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ============================================================
        #  Compute a search direction pk by applying the CG method to
        #       H_f(xk) p = - J_f(xk) starting from 0.
        # ============================================================

        # Compute search direction with conjugate gradient (GG)
        d, cg_iters, cg_fail = _cg_iters(g, hessp, cg_max_iter, normp)
        ncg += cg_iters

        if cg_fail:
            warnflag = 3
            msg = _status_message['cg_warn']
            break

        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == 'none':
            update = d.mul(lr)
            x = x + update
        elif line_search == 'strong-wolfe':
            # strong-wolfe line search
            _, _, t, ls_nevals = strong_wolfe(dir_evaluate, x, lr, d, f, g)
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        # re-evaluate function
        f, g, hessp, _ = closure(x)

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (n_iter, f))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)

        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=normp) <= xtol:
            warnflag = 0
            msg = _status_message['success']
            break

        if not f.isfinite():
            warnflag = 3
            msg = _status_message['nan']
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1
        msg = _status_message['maxiter']

    if disp:
        print(msg)
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
        print("         CG iterations: %d" % ncg)
    result = OptimizeResult(fun=f, x=x.view_as(x0), grad=g.view_as(x0),
                            status=warnflag, success=(warnflag==0),
                            message=msg, nit=n_iter, nfev=sf.nfev, ncg=ncg)
    if return_all:
        result['allvecs'] = allvecs
    return result



@torch.no_grad()
def _minimize_newton_exact(
        fun, x0, lr=1., max_iter=None, line_search='strong-wolfe', xtol=1e-5,
        normp=1, tikhonov=0., handle_npd='grad', callback=None, disp=0,
        return_all=False):
    """Minimize a scalar function of one or more variables using the
    Newton-Raphson method.

    This variant uses an "exact" Newton routine based on Cholesky factorization
    of the explicit Hessian matrix.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    tikhonov : float
        Optional diagonal regularization (Tikhonov) parameter for the Hessian.
    handle_npd : str
        Mode for handling non-positive definite hessian matrices. Can be one
        of the following:

            * 'grad' : use steepest descent direction (gradient)
            * 'lu' : solve the inverse hessian with LU factorization
            * 'eig' : use symmetric eigendecomposition to determine a
              diagonal regularization parameter
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
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
    sf = ScalarFunction(fun, x0.shape, hess=True)
    closure = sf.closure
    if line_search == 'strong-wolfe':
        dir_evaluate = sf.dir_evaluate

    # initial settings
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    f, g, _, hess = closure(x)
    if tikhonov > 0:
        hess.diagonal().add_(tikhonov)
    if disp > 1:
        print('initial fval: %0.4f' % f)
    if return_all:
        allvecs = [x]
    nfail = 0
    n_iter = 0

    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ==================================================
        #  Compute a search direction d by solving
        #          H_f(x) d = - J_f(x)
        #  with the true Hessian and Cholesky factorization
        # ===================================================

        # Compute search direction with Cholesky solve
        L, info = torch.linalg.cholesky_ex(hess)

        if info == 0:
            d = torch.cholesky_solve(g.neg().unsqueeze(1), L).squeeze(1)
        else:
            nfail += 1
            if handle_npd == 'lu':
                d = torch.linalg.solve(hess, g.neg())
            elif handle_npd in ['grad', 'cauchy']:
                d = g.neg()
                if handle_npd == 'cauchy':
                    # cauchy point for a trust radius of delta=1.
                    # equivalent to 'grad' with a scaled lr
                    gnorm = g.norm(p=2)
                    scale = 1 / gnorm
                    gHg = g.dot(hess.mv(g))
                    if gHg > 0:
                        scale *= torch.clamp_(gnorm.pow(3) / gHg, max=1)
                    d *= scale
            elif handle_npd == 'eig':
                # this setting is experimental! use with caution
                # TODO: why use the factor 1.5 here? Seems to work best
                eig0 = eigsh(hess.cpu().numpy(), k=1, which="SA", tol=1e-4)[0].item()
                tau = max(1e-3 - 1.5 * eig0, 0)
                hess.diagonal().add_(tau)
                L = torch.linalg.cholesky(hess)
                d = torch.cholesky_solve(g.neg().unsqueeze(1), L).squeeze(1)
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
            _, _, t, ls_nevals = strong_wolfe(dir_evaluate, x, lr, d, f, g)
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        # ===================================
        #  Re-evaluate func/Jacobian/Hessian
        # ===================================

        f, g, _, hess = closure(x)
        if tikhonov > 0:
            hess.diagonal().add_(tikhonov)

        if disp > 1:
            print('iter %3d - fval: %0.4f - info: %d' % (n_iter, f, info))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)

        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=normp) <= xtol:
            warnflag = 0
            msg = _status_message['success']
            break

        if not f.isfinite():
            warnflag = 3
            msg = _status_message['nan']
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1
        msg = _status_message['maxiter']

    if disp:
        print(msg)
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
    result = OptimizeResult(fun=f, x=x.view_as(x0), grad=g.view_as(x0),
                            hess=hess.view(2 * x0.shape),
                            status=warnflag, success=(warnflag==0),
                            message=msg, nit=n_iter, nfev=sf.nfev, nfail=nfail)
    if return_all:
        result['allvecs'] = allvecs
    return result

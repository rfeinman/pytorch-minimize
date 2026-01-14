import numpy as np
import torch
from torch import Tensor
from scipy.optimize import (
    linear_sum_assignment,
    OptimizeResult,
)
from scipy.sparse.linalg import svds

from .function import ScalarFunction

try:
    from scipy.optimize.optimize import _status_message
except ImportError:
    from scipy.optimize._optimize import _status_message


@torch.no_grad()
def _minimize_constr_tracenorm(
        fun, x0, t, max_iter=None, gtol=1e-5, normp=float('inf'),
        callback=None, disp=0):
    """Minimize a scalar function of matrix, constrained to have trace-norm
    (i.e. nuclear norm) less than t.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    t : float
        Maximum allowed trace norm.
    max_iter : int, optional
        Maximum number of iterations to perform.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    normp : float
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    m, n = x0.shape

    disp = int(disp)
    if max_iter is None:
        max_iter = m * 100

    sf = ScalarFunction(fun, x_shape=x0.shape)
    closure = sf.closure
    dir_evaluate = sf.dir_evaluate
    x = x0.detach()
    for niter in range(max_iter):
        f, g, _, _ = closure(x)
        [u, s, vh] = svds(g.detach().numpy(), k=1)
        alpha = 2. / (niter + 2.)
        x = (1 - alpha) * x + alpha * Tensor(-t * u @ vh)
        if disp > 1:
            print('iter %3d - fval: %0.4f' % (niter, f))

        if callback is not None:
            callback(x)

        # check optimality
        grad_norm = g.norm(p=normp)
        if grad_norm <= gtol:
            warnflag = 0
            msg = _status_message['success']
            break

    else:
        # if we get to the end, the maximum iterations was reached
        warnflag = 1
        msg = _status_message['maxiter']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % niter)
        print("         Function evaluations: %d" % sf.nfev)

    result = OptimizeResult(fun=f, x=x.view_as(x0), grad=g.view_as(x0),
                            status=warnflag, success=(warnflag == 0),
                            message=msg, nit=niter, nfev=sf.nfev)
    return result


@torch.no_grad()
def _minimize_constr_birkhoff_polytope(
        fun, x0, max_iter=None, gtol=1e-5, normp=float('inf'),
        callback=None, disp=0):
    """Minimize a scalar function of a square matrix, constrained to lie in
    the Birkhoff polytope, i.e. over the space of doubly stochastic matrices.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    max_iter : int, optional
        Maximum number of iterations to perform.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    normp : float
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    m, n = x0.shape
    
    if m != n:
        raise RuntimeError('Initial iterate must be a square matrix.')
    
    if not ((x0.sum(0) == 1).all() and (x0.sum(1) == 1).all()):
        raise RuntimeError('Initial iterate must be doubly stochastic.')

    disp = int(disp)
    if max_iter is None:
        max_iter = m * 100

    # Construct scalar objective function
    sf = ScalarFunction(fun, x_shape=x0.shape)
    closure = sf.closure
    dir_evaluate = sf.dir_evaluate
    x = x0.detach()
    for niter in range(max_iter):
        f, g, _, _ = closure(x)
        [row_ind, col_ind] = linear_sum_assignment(g.detach().numpy())
        alpha = 2. / (niter + 2.)
        x = (1 - alpha) * x
        x[row_ind, col_ind] += alpha
        if disp > 1:
            print('iter %3d - fval: %0.4f' % (niter, f))

        if callback is not None:
            callback(x)

        # check optimality
        grad_norm = g.norm(p=normp)
        if grad_norm <= gtol:
            warnflag = 0
            msg = _status_message['success']
            break

    else:
        # if we get to the end, the maximum iterations was reached
        warnflag = 1
        msg = _status_message['maxiter']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % niter)
        print("         Function evaluations: %d" % sf.nfev)

    result = OptimizeResult(fun=f, x=x.view_as(x0), grad=g.view_as(x0),
                            status=warnflag, success=(warnflag == 0),
                            message=msg, nit=niter, nfev=sf.nfev)
    return result

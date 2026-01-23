import warnings
import numpy as np
import torch
from numbers import Number
from scipy.optimize import (
    linear_sum_assignment,
    OptimizeResult,
)
from scipy.sparse.linalg import svds

from .._optimize import _status_message
from ..function import ScalarFunction


@torch.no_grad()
def _minimize_frankwolfe(
        fun, x0, constr='tracenorm', t=None, max_iter=None, gtol=1e-5,
        normp=float('inf'), callback=None, disp=0):
    """Minimize a scalar function of a matrix with Frank-Wolfe (a.k.a.
    conditional gradient).

    The algorithm is described in [1]_. The following constraints are currently 
    supported:

        - Trace norm. The matrix is constrained to have trace norm (a.k.a.
          nuclear norm) less than t.
        - Birkhoff polytope. The matrix is constrained to lie in the Birkhoff
          polytope, i.e. over the space of doubly stochastic matrices. Requires
          a square matrix.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    constr : str
        Which constraint to use. Must be either 'tracenorm' or 'birkhoff'.
    t : float, optional
        Maximum allowed trace norm. Required when using the 'tracenorm' constr;
        otherwise unused.
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

    References
    ----------
    .. [1] Martin Jaggi, "Revisiting Frank-Wolfe: Projection-Free Sparse Convex
       Optimization", ICML 2013.

    """
    assert isinstance(constr, str)
    constr = constr.lower()
    if constr in {'tracenorm', 'trace-norm'}:
        assert t is not None, \
            f'Argument `t` is required when using the trace-norm constraint.'
        assert isinstance(t, Number), \
            f'Argument `t` must be a Number but got {type(t)}'
        constr = 'tracenorm'
    elif constr in {'birkhoff', 'birkhoff-polytope'}:
        if t is not None:
            warnings.warn(
                'Argument `t` was provided but is unused for the'
                'birkhoff-polytope constraint.'
            )
        constr = 'birkhoff'
    else:
        raise ValueError(f'Invalid constr: "{constr}".')

    if x0.ndim != 2:
        raise ValueError(
            f'Optimization variable `x` must be a matrix to use Frank-Wolfe.'
        )

    m, n = x0.shape

    if constr == 'birkhoff':
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

        if constr == 'tracenorm':
            u, s, vh = svds(g.detach().numpy(), k=1)
            uvh = x.new_tensor(u @ vh)
            alpha = 2. / (niter + 2.)
            x = torch.lerp(x, -t * uvh, weight=alpha)
        elif constr == 'birkhoff':
            row_ind, col_ind = linear_sum_assignment(g.detach().numpy())
            alpha = 2. / (niter + 2.)
            x = (1 - alpha) * x
            x[row_ind, col_ind] += alpha
        else:
            raise ValueError

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (niter, f))

        if callback is not None:
            if callback(x):
                warnflag = 5
                msg = _status_message['callback_stop']
                break

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

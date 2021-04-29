import warnings
import torch
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message

from .function import ScalarFunction
from .line_search import strong_wolfe


@torch.no_grad()
def _minimize_cg(fun, x0, max_iter=None, gtol=1e-5, normp=float('inf'),
                 callback=None, disp=0, return_all=False):
    """Minimize a scalar function of one or more variables using the
    conjugate gradient algorithm.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    max_iter : int
        Maximum number of iterations to perform. Defaults to 200 * x0.numel()
    gtol : float
        Termination tolerance on 1st-order optimality (gradient magnitude)
    normp : float
        The norm type to use for termination conditions. Can be any value
        supported by ``torch.norm``.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. callback(x_k)
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    """
    disp = int(disp)
    if max_iter is None:
        max_iter = x0.numel() * 200

    # Construct scalar objective function
    sf = ScalarFunction(fun, x_shape=x0.shape)
    closure = sf.closure
    dir_evaluate = sf.dir_evaluate

    # initialize
    x = x0.detach().flatten()
    f, g, _, _ = closure(x)
    if disp > 1:
        print('initial fval: %0.4f' % f)
    if return_all:
        allvecs = [x]
    d = g.neg()
    grad_norm = g.norm(p=normp)
    old_f = f + g.norm() / 2  # Sets the initial step guess to dx ~ 1

    for niter in range(1, max_iter + 1):
        delta = g.dot(g)

        cached_step = [None]

        def polak_ribiere_powell_step(t, g_next):
            x_next = x + d.mul(t)
            y = g_next - g
            beta = torch.clamp(y.dot(g_next) / delta, min=0)
            d_next = -g_next + d.mul(beta)
            torch.norm(g_next, p=normp, out=grad_norm)
            return t, x_next, d_next, g_next

        def descent_condition(t, f_next, g_next):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            cached_step[:] = polak_ribiere_powell_step(t, g_next)
            t, x, d, g = cached_step

            # Accept step if it leads to convergence or if sufficient
            # descent condition applies.
            return (grad_norm <= gtol) | (d.dot(g) <= -0.01 * g.dot(g))

        gtd = g.dot(d)
        t0 = torch.clamp(2.02 * (f - old_f) / gtd, max=1.0)
        if t0 <= 0:
            warnings.warn('initial step guess is negative.')

        old_f = f
        f, g_next, t, ls_evals = \
                strong_wolfe(dir_evaluate, x, t0, d, f, g, gtd,
                             c2=0.4, extra_condition=descent_condition)

        # Reuse already computed results if possible
        if t == cached_step[0]:
            t, x, d, g = cached_step
        else:
            t, x, d, g = polak_ribiere_powell_step(t, g_next)

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (niter, f))
        if return_all:
            allvecs.append(x)
        if callback is not None:
            callback(x)

        # check optimality
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

    result = OptimizeResult(x=x, fun=f, jac=g, nit=niter,  nfev=sf.nfev,
                            status=warnflag, success=(warnflag == 0),
                            message=msg)
    if return_all:
        result['allvecs'] = allvecs
    return result
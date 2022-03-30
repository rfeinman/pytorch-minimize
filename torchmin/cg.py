import torch
from scipy.optimize import OptimizeResult

from .function import ScalarFunction
from .line_search import strong_wolfe

try:
    from scipy.optimize.optimize import _status_message
except ImportError:
    from scipy.optimize._optimize import _status_message

dot = lambda u,v: torch.dot(u.view(-1), v.view(-1))


@torch.no_grad()
def _minimize_cg(fun, x0, max_iter=None, gtol=1e-5, normp=float('inf'),
                 callback=None, disp=0, return_all=False):
    """Minimize a scalar function of one or more variables using
    nonlinear conjugate gradient.

    The algorithm is described in Nocedal & Wright (2006) chapter 5.2.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    max_iter : int
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    normp : float
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``
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
        # delta/gtd
        delta = dot(g, g)
        gtd = dot(g, d)

        # compute initial step guess based on (f - old_f) / gtd
        t0 = torch.clamp(2.02 * (f - old_f) / gtd, max=1.0)
        if t0 <= 0:
            warnflag = 4
            msg = 'Initial step guess is negative.'
            break
        old_f = f

        # buffer to store next direction vector
        cached_step = [None]

        def polak_ribiere_powell_step(t, g_next):
            y = g_next - g
            beta = torch.clamp(dot(y, g_next) / delta, min=0)
            d_next = -g_next + d.mul(beta)
            torch.norm(g_next, p=normp, out=grad_norm)
            return t, d_next

        def descent_condition(t, f_next, g_next):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            cached_step[:] = polak_ribiere_powell_step(t, g_next)
            t, d_next = cached_step

            # Accept step if it leads to convergence.
            cond1 = grad_norm <= gtol

            # Accept step if sufficient descent condition applies.
            cond2 = dot(d_next, g_next) <= -0.01 * dot(g_next, g_next)

            return cond1 | cond2

        # Perform CG step
        f, g, t, ls_evals = \
                strong_wolfe(dir_evaluate, x, t0, d, f, g, gtd,
                             c2=0.4, extra_condition=descent_condition)

        # Update x and then update d (in that order)
        x = x + d.mul(t)
        if t == cached_step[0]:
            # Reuse already computed results if possible
            d = cached_step[1]
        else:
            d = polak_ribiere_powell_step(t, g)[1]

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

    result = OptimizeResult(fun=f, x=x.view_as(x0), grad=g.view_as(x0),
                            status=warnflag, success=(warnflag == 0),
                            message=msg, nit=niter, nfev=sf.nfev)
    if return_all:
        result['allvecs'] = allvecs
    return result
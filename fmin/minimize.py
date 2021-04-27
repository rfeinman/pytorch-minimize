import torch

from .bfgs import _minimize_bfgs
from .newton import _minimize_newton_cg, _minimize_newton_exact
from .trustregion import (_minimize_trust_exact, _minimize_dogleg,
                          _minimize_trust_ncg)

_tolerance_keys = {
    'l-bfgs': 'gtol',
    'bfgs': 'gtol',
    'newton-cg': 'xtol',
    'newton-exact': 'xtol',
    'dogleg': 'gtol',
    'trust-ncg': 'gtol',
    'trust-exact': 'gtol'
}


def minimize(
        f, x0, method, max_iter=None, tol=None, options=None, callback=None,
        disp=0, return_all=False):
    """Minimize a scalar function of one or more variables.

    .. note::
        This is a general-purpose minimizer that calls one of the available
        routines based on a supplied `method` argument.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    method : str
        The minimization routine to use. Should be one of

            - 'bfgs'
            - 'l-bfgs'
            - 'newton-cg'
            - 'newton-exact'
            - 'dogleg'
            - 'trust-ncg'
            - 'trust-exact'

        At the moment, method must be specified; there is no default.
    max_iter : int, optional
        Maximum number of iterations to perform. If unspecified, this will
        be set to the default of the selected method.
    tol : float
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of keyword arguments to pass to the selected minimization
        routine.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    x0 = torch.as_tensor(x0)
    method = method.lower()
    assert method in ['bfgs', 'l-bfgs', 'newton-cg', 'newton-exact',
                      'dogleg', 'trust-ncg', 'trust-exact']
    if options is None:
        options = {}
    if tol is not None:
        options.setdefault(_tolerance_keys[method], tol)
    options.setdefault('max_iter', max_iter)
    options.setdefault('callback', callback)
    options.setdefault('disp', disp)
    options.setdefault('return_all', return_all)

    if method in ['bfgs', 'l-bfgs']:
        options.setdefault('low_mem', method == 'l-bfgs')
        return _minimize_bfgs(f, x0, **options)
    elif method == 'newton-cg':
        return _minimize_newton_cg(f, x0, **options)
    elif method == 'newton-exact':
        return _minimize_newton_exact(f, x0, **options)
    elif method == 'dogleg':
        return _minimize_dogleg(f, x0, **options)
    elif method == 'trust-ncg':
        return _minimize_trust_ncg(f, x0, **options)
    elif method == 'trust-exact':
        return _minimize_trust_exact(f, x0, **options)
    else:
        raise RuntimeError('invalid method "{}" encountered.'.format(method))
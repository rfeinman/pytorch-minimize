import torch

from .constrained.lbfgsb import _minimize_lbfgsb
from .constrained.frankwolfe import _minimize_frankwolfe
from .constrained.trust_constr import _minimize_trust_constr


_tolerance_keys = {
    'l-bfgs-b': 'gtol',
    'frank-wolfe': 'gtol',
    'trust-constr': 'tol',
}


def minimize_constr(
        f,
        x0,
        method=None,
        constr=None,
        bounds=None,
        max_iter=None,
        tol=None,
        options=None,
        callback=None,
        disp=0,
        ):
    """Minimize a scalar function of one or more variables subject to
    bounds and/or constraints.

    .. note::
        The `trust-constr` method is currently a wrapper for
        `SciPy's trustconstr <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_
        implementation.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    method : str, optional
        The minimization routine to use. Should be one of the following:

            - 'l-bfgs-b'
            - 'tracenorm'
            - 'birkhoff_polytope'
            - 'trust-constr'

        If no method is provided, a default method will be selected based
        on the criteria of the problem.
    constr : dict or string, optional
        Constraint specifications. Should either be a string (Frank-Wolfe
        method) or a dictionary (trust-constr method) with the following fields:

            * fun (callable) - Constraint function
            * lb (Tensor or float, optional) - Constraint lower bounds
            * ub (Tensor or float, optional) - Constraint upper bounds

        One of either `lb` or `ub` must be provided. When `lb` == `ub` it is
        interpreted as an equality constraint.
    bounds : dict, optional
        **TODO: update bounds convention & argument documentation.**
        Bounds on variables. Should be a dictionary with at least one
        of the following fields:

            * lb (Tensor or float) - Lower bounds
            * ub (Tensor or float) - Upper bounds

        Bounds of `-inf`/`inf` are interpreted as no bound. When `lb` == `ub`
        it is interpreted as an equality constraint.
    max_iter : int, optional
        Maximum number of iterations to perform. If unspecified, this will
        be set to the default of the selected method.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int
        Level of algorithm's verbosity:

            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).
    **kwargs
        Additional keyword arguments passed to SciPy's trust-constr solver.
        See options `here <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """

    if method is None:
        if constr is not None:
            _frank_wolfe_constraints = {
                'tracenorm', 'trace-norm', 'birkhoff', 'birkhoff-polytope'}
            if (
                isinstance(constr, str)
                and constr.lower() in _frank_wolfe_constraints
                ):
                method = 'frank-wolfe'
            else:
                method = 'trust-constr'
        else:
            method = 'l-bfgs-b'

    assert isinstance(method, str)
    method = method.lower()

    if options is None:
        options = {}
    else:
        assert isinstance(options, dict)
        options = options.copy()
    options.setdefault('max_iter', max_iter)
    options.setdefault('callback', callback)
    options.setdefault('disp', disp)
    # options.setdefault('return_all', return_all)
    if tol is not None:
        options.setdefault(_tolerance_keys[method], tol)

    if method == 'l-bfgs-b':
        assert constr is None
        return _minimize_lbfgsb(f, x0, bounds=bounds, **options)
    elif method == 'frank-wolfe':
        assert bounds is None
        return _minimize_frankwolfe(f, x0, constr=constr, **options)
    elif method == 'trust-constr':
        return _minimize_trust_constr(
            f, x0, constr=constr, bounds=bounds, **options)
    else:
        raise RuntimeError(f'Invalid method: "{method}".')

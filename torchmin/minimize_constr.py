import numbers
import numpy as np
import torch
from scipy.optimize import Bounds

from .constrained.lbfgsb import _minimize_lbfgsb
from .constrained.frankwolfe import _minimize_frankwolfe
from .constrained.trust_constr import _minimize_trust_constr


_tolerance_keys = {
    'l-bfgs-b': 'gtol',
    'frank-wolfe': 'gtol',
    'trust-constr': 'tol',
}


def _maybe_to_number(val):
    if isinstance(val, np.ndarray) and val.size == 1:
        return val.item()
    elif isinstance(val, torch.Tensor) and val.numel() == 1:
        return val.item()
    else:
        return val


def _check_bound(val, x0, numpy=False):
    n = x0.numel()
    if isinstance(val, numbers.Number):
        if numpy:
            return np.full(n, val, dtype=float)  # TODO: correct dtype
        else:
            return x0.new_full((n,), val)

    if isinstance(val, (list, tuple)):
        if numpy:
            val = np.array(val, dtype=float)  # TODO: correct dtype
        else:
            val = x0.new_tensor(val)

    if isinstance(val, torch.Tensor):
        assert val.numel() == n, f'Bound tensor has incorrect size'
        val = val.flatten()
        if numpy:
            val = val.detach().cpu().numpy()
        return val
    elif isinstance(val, np.ndarray):
        assert val.size == n, f'Bound array has incorrect size'
        val = val.flatten()
        if not numpy:
            val = x0.new_tensor(val)
        return val
    else:
        raise ValueError(f'Bound has invalid type: {type(val)}')


def _check_bounds(bounds, x0, method):
    if isinstance(bounds, Bounds):
        if method == 'trust-constr':
            return bounds
        else:
            bounds = (bounds.lb, bounds.ub)
            bounds = tuple(map(_maybe_to_number, bounds))

    assert isinstance(bounds, (list, tuple)), \
        f'Argument `bounds` must be a list or tuple but got {type(bounds)}'
    assert len(bounds) == 2, \
        f'Argument `bounds` must have length 2: (min, max)'
    lb, ub = bounds

    lb = float('-inf') if lb is None else lb
    ub = float('inf') if ub is None else ub

    numpy = (method == 'trust-constr')
    lb = _check_bound(lb, x0, numpy=numpy)
    ub = _check_bound(ub, x0, numpy=numpy)

    return lb, ub


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
        Method ``'trust-constr'`` is currently a wrapper for SciPy's 
        `trust-constr <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_ 
        solver.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    method : str, optional
        The minimization routine to use. Should be one of the following:

            - 'l-bfgs-b'
            - 'frank-wolfe'
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
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:

            1. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
            2. Instance of :class:`scipy.optimize.Bounds` class.

        Bounds of `-inf`/`inf` are interpreted as no bound. When `lb` == `ub`
        it is interpreted as an equality constraint.
    max_iter : int, optional
        Maximum number of iterations to perform. If unspecified, this will
        be set to the default of the selected method.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of keyword arguments to pass to the selected minimization
        routine.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int
        Level of algorithm's verbosity:

            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).

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

    if bounds is not None:
        bounds = _check_bounds(bounds, x0, method)

        # TODO: update `_minimize_trust_constr()` accepted bounds format
        # and remove this
        if method == 'trust-constr':
            if isinstance(bounds, Bounds):
                bounds = dict(
                    lb=_maybe_to_number(bounds.lb),
                    ub=_maybe_to_number(bounds.ub),
                    keep_feasible=bounds.keep_feasible,
                )
            else:
                bounds = dict(lb=bounds[0], ub=bounds[1])

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

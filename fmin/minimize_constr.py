import warnings
import numbers
import torch
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.sparse.linalg import LinearOperator

_constr_keys = {'fun', 'lb', 'ub', 'jac', 'hess', 'hessp', 'keep_feasible'}
_bounds_keys = {'lb', 'ub', 'keep_feasible'}


def _build_obj(f, x0):
    numel = x0.numel()

    def to_tensor(x):
        return torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)

    def f_with_jac(x):
        x = to_tensor(x).requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        grad, = torch.autograd.grad(fval, x)
        return fval.detach().cpu().numpy(), grad.view(-1).cpu().numpy()

    def f_hess(x):
        x = to_tensor(x).requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
            grad, = torch.autograd.grad(fval, x, create_graph=True)
        def matvec(p):
            p = to_tensor(p)
            hvp, = torch.autograd.grad(grad, x, p, retain_graph=True)
            return hvp.view(-1).cpu().numpy()
        return LinearOperator((numel, numel), matvec=matvec)

    return f_with_jac, f_hess


def _build_constr(constr, x0):
    assert isinstance(constr, dict)
    assert set(constr.keys()).issubset(_constr_keys)
    assert 'fun' in constr
    assert 'lb' in constr or 'ub' in constr
    if 'lb' not in constr:
        constr['lb'] = -np.inf
    if 'ub' not in constr:
        constr['ub'] = np.inf
    f_ = constr['fun']
    numel = x0.numel()

    def to_tensor(x):
        return torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)

    def f(x):
        x = to_tensor(x)
        return f_(x).cpu().numpy()

    def f_jac(x):
        x = to_tensor(x)
        if 'jac' in constr:
            grad = constr['jac'](x)
        else:
            x.requires_grad_(True)
            with torch.enable_grad():
                grad, = torch.autograd.grad(f_(x), x)
        return grad.view(-1).cpu().numpy()

    def f_hess(x, v):
        x = to_tensor(x)
        if 'hess' in constr:
            hess = constr['hess'](x)
            return v[0] * hess.view(numel, numel).cpu().numpy()
        elif 'hessp' in constr:
            def matvec(p):
                p = to_tensor(p)
                hvp = constr['hessp'](x, p)
                return v[0] * hvp.view(-1).cpu().numpy()
            return LinearOperator((numel, numel), matvec=matvec)
        else:
            x.requires_grad_(True)
            with torch.enable_grad():
                if 'jac' in constr:
                    grad = constr['jac'](x)
                else:
                    grad, = torch.autograd.grad(f_(x), x, create_graph=True)
            def matvec(p):
                p = to_tensor(p)
                hvp, = torch.autograd.grad(grad, x, p, retain_graph=True)
                return v[0] * hvp.view(-1).cpu().numpy()
            return LinearOperator((numel, numel), matvec=matvec)

    return NonlinearConstraint(
        fun=f, lb=constr['lb'], ub=constr['ub'],
        jac=f_jac, hess=f_hess,
        keep_feasible=constr.get('keep_feasible', False))


def _check_bound(val, x0):
    if isinstance(val, numbers.Number):
        return np.full(x0.numel(), val)
    elif isinstance(val, torch.Tensor):
        assert val.numel() == x0.numel()
        return val.detach().cpu().numpy().flatten()
    elif isinstance(val, np.ndarray):
        assert val.size == x0.numel()
        return val.flatten()
    else:
        raise ValueError('Bound value has unrecognized format.')


def _build_bounds(bounds, x0):
    assert isinstance(bounds, dict)
    assert set(bounds.keys()).issubset(_bounds_keys)
    assert 'lb' in bounds or 'ub' in bounds
    lb = _check_bound(bounds.get('lb', -np.inf), x0)
    ub = _check_bound(bounds.get('ub', np.inf), x0)
    keep_feasible = bounds.get('keep_feasible', False)

    return Bounds(lb, ub, keep_feasible)


@torch.no_grad()
def minimize_constr(
        f, x0, constr=None, bounds=None, max_iter=None, tol=None, callback=None,
        disp=0, **kwargs):
    """Minimize a scalar function of one or more variables subject to
    bounds and/or constraints.

    .. note::
        This is a wrapper for SciPy's
        `'trust-constr' <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_
        method. It uses autograd behind the scenes to build jacobian & hessian
        callables before invoking scipy. Inputs and objectivs should use
        PyTorch tensors like other routines. CUDA is supported; however,
        data will be transferred back-and-forth between GPU/CPU.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    constr : dict, optional
        Constraint specifications. Should be a dictionary with the
        following fields:

            * fun (callable) - Constraint function
            * lb (Tensor or float, optional) - Constraint lower bounds
            * ub : (Tensor or float, optional) - Constraint upper bounds

        One of either `lb` or `ub` must be provided. When `lb` == `ub` it is
        interpreted as an equality constraint.
    bounds : dict, optional
        Bounds on variables. Should a dictionary with at least one
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
    if max_iter is None:
        max_iter = 1000
    x0 = x0.detach()
    if x0.is_cuda:
        warnings.warn('GPU is not recommended for trust-constr. '
                      'Data will be moved back-and-forth from CPU.')

    # handle callbacks
    if callback is not None:
        callback_ = callback
        callback = lambda x: callback_(
            torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0))

    # handle bounds
    if bounds is not None:
        bounds = _build_bounds(bounds, x0)

    # build objective function (and hessian)
    f_with_jac, f_hess = _build_obj(f, x0)

    # build constraints
    if constr is not None:
        constraints = [_build_constr(constr, x0)]
    else:
        constraints = []

    # optimize
    x0_np = x0.cpu().numpy().flatten().copy()
    result = minimize(
        f_with_jac, x0_np, method='trust-constr', jac=True,
        hess=f_hess, callback=callback, tol=tol,
        bounds=bounds,
        constraints=constraints,
        options=dict(verbose=int(disp), maxiter=max_iter, **kwargs)
    )

    # convert the important things to torch tensors
    for key in ['fun', 'grad', 'x']:
        result[key] = torch.tensor(result[key], dtype=x0.dtype, device=x0.device)
    result['x'] = result['x'].view_as(x0)

    return result


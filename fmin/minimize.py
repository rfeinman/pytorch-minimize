import torch

from .bfgs import fmin_bfgs
from .newton import fmin_newton_cg, fmin_newton_exact

_tolerance_keys = {
    'bfgs': 'gtol',
    'newton_cg': 'xtol',
    'newton_exact': 'xtol'
}

def minimize(
        f, x0, method, max_iter=None, tol=None, options=None, callback=None,
        disp=0, return_all=False):
    """A general-purpose minimization routine that calls one of the available
    algorithms based on a "method" argument.
    """

    x0 = torch.as_tensor(x0)
    method = method.lower()
    assert method in ['bfgs', 'newton_cg', 'newton_exact']
    if options is None:
        options = {}
    if tol is None:
        options[_tolerance_keys[method]] = tol
    options['max_iter'] = max_iter
    options['callback'] = callback
    options['disp'] = disp
    options['return_all'] = return_all

    if method == 'bfgs':
        return fmin_bfgs(f, x0, **options)
    elif method == 'newton_cg':
        return fmin_newton_cg(f, x0, **options)
    elif method == 'newton_exact':
        return fmin_newton_exact(f, x0, **options)
    else:
        raise RuntimeError('invalid method "{}" encountered.'.format(method))
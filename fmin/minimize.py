import warnings
import torch

from .bfgs import fmin_bfgs
from .newton import fmin_newton_cg, fmin_newton_exact

_tolerance_keys = {
    'bfgs': 'gtol',
    'newton-cg': 'xtol',
    'newton-exact': 'xtol'
}

def minimize(
        f, x0, method, max_iter=None, tol=None, options=None, callback=None,
        disp=0, return_all=False):
    """A general-purpose minimization routine that calls one of the available
    algorithms based on a "method" argument.
    """

    x0 = torch.as_tensor(x0)
    method = method.lower()
    assert method in ['bfgs', 'l-bfgs', 'newton-cg', 'newton-exact']
    if options is None:
        options = {}
    if tol is not None:
        options.setdefault(_tolerance_keys[method], tol)
    options.setdefault('max_iter', max_iter)
    options.setdefault('callback', callback)
    options.setdefault('disp', disp)
    options.setdefault('return_all', return_all)

    if method in ['bfgs', 'l-bfgs']:
        if method == 'bfgs' and options.get('low_mem', False):
            warnings.warn("Usage {method='bfgs', low_mem=True} is "
                          "not recommended. Use {method='l-bfgs'} instead.")
            method = 'l-bfgs'
        options['low_mem'] = method == 'l-bfgs'
        return fmin_bfgs(f, x0, **options)
    elif method == 'newton-cg':
        return fmin_newton_cg(f, x0, **options)
    elif method == 'newton-exact':
        return fmin_newton_exact(f, x0, **options)
    else:
        raise RuntimeError('invalid method "{}" encountered.'.format(method))
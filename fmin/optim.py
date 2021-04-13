import numbers
import numpy as np
import torch
from functools import reduce
from torch.optim import Optimizer
from scipy import optimize


def _build_bounds(bounds, params, numel_total):
    if len(bounds) != len(params):
        raise ValueError('bounds must be an iterable with same length as params')

    lb = np.full(numel_total, -np.inf)
    ub = np.full(numel_total, np.inf)
    keep_feasible = np.zeros(numel_total, dtype=np.bool)

    def process_bound(x, numel):
        if isinstance(x, torch.Tensor):
            assert x.numel() == numel
            return x.view(-1).detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            assert x.size == numel
            return x.flatten()
        elif isinstance(x, (bool, numbers.Number)):
            return x
        else:
            raise ValueError('invalid bound value.')

    offset = 0
    for bound, p in zip(bounds, params):
        numel = p.numel()
        if bound is None:
            offset += numel
            continue
        if not isinstance(bound, (list, tuple)) and len(bound) in [2,3]:
            raise ValueError('elements of "bounds" must each be a '
                             'list/tuple of length 2 or 3')
        if bound[0] is None and bound[1] is None:
            raise ValueError('either lower or upper bound must be defined.')
        if bound[0] is not None:
            lb[offset:offset + numel] = process_bound(bound[0], numel)
        if bound[1] is not None:
            ub[offset:offset + numel] = process_bound(bound[1], numel)
        if len(bound) == 3:
            keep_feasible[offset:offset + numel] = process_bound(bound[2], numel)
        offset += numel

    return optimize.Bounds(lb, ub, keep_feasible)


class Minimize(Optimizer):
    """A general-purpose optimizer that wraps SciPy's `optimize.minimize`.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Args:
        method (str or callable, optional): TODO
        bounds (): TODO
        constraints (): TODO
        tol (float, optional): TODO
        options (dict, optional): TODO

    """
    def __init__(self,
                 params,
                 method='bfgs',
                 bounds=None,
                 constraints=(),
                 tol=None,
                 options=None):
        assert isinstance(method, str)
        method = method.lower()
        defaults = dict(
            method=method,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Minimize doesn't support per-parameter options "
                             "(parameter groups)")
        if constraints != ():
            raise NotImplementedError("Minimize doesn't yet support constraints")

        self._params = self.param_groups[0]['params']
        self._param_bounds = self.param_groups[0]['bounds']
        self._numel_cache = None
        self._bounds_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _bounds(self):
        if self._param_bounds is None:
            return None
        if self._bounds_cache is None:
            self._bounds_cache = _build_bounds(self._param_bounds, self._params,
                                               self._numel())
        return self._bounds_cache

    def _gather_flat_param(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _set_flat_param(self, value):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.copy_(value[offset:offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        # sanity check
        assert len(self.param_groups) == 1

        # functions to convert numpy -> torch and torch -> numpy
        to_tensor = lambda x: self._params[0].new_tensor(x)
        to_array = lambda x: x.cpu().numpy()

        # optimizer settings
        group = self.param_groups[0]
        method = group['method']
        bounds = self._bounds()
        constraints = group['constraints']
        tol = group['tol']
        options = group['options']

        # build objective
        def fun(x):
            x = to_tensor(x)
            self._set_flat_param(x)
            with torch.enable_grad():
                loss = closure()
            grad = self._gather_flat_grad()
            return float(loss), to_array(grad)

        # initial value (numpy array)
        x0 = to_array(self._gather_flat_param())

        # optimize
        result = optimize.minimize(
            fun, x0, method=method, jac=True, bounds=bounds,
            constraints=constraints, tol=tol, options=options
        )

        # set final param
        self._set_flat_param(to_tensor(result.x))

        return result

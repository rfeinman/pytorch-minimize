import torch
from functools import reduce
from torch.optim import Optimizer
from scipy import optimize


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
        if bounds != None:
            raise NotImplementedError("Minimize doesn't yet support bounds")
        if constraints != ():
            raise NotImplementedError("Minimize doesn't yet support constraints")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

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
        bounds = group['bounds']
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

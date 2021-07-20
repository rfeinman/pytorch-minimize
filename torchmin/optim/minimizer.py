from functools import reduce
import torch
from torch.optim import Optimizer


class LinearOperator:
    """A generic linear operator to use with Minimizer"""
    def __init__(self, matvec, shape, dtype=torch.float, device=None):
        self.rmv = matvec
        self.mv = matvec
        self.shape = shape
        self.dtype = dtype
        self.device = device


class Minimizer(Optimizer):
    """A general-purpose PyTorch optimizer for unconstrained function
    minimization.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Parameters
    ----------
    params : iterable
        An iterable of :class:`torch.Tensor` s. Specifies what Tensors
        should be optimized.
    method : str
        Minimization method (algorithm) to use. Must be one of the methods
        offered in :func:`torchmin.minimize()`. Defaults to 'bfgs'.
    **minimize_kwargs : dict
        Additional keyword arguments that will be passed to
        :func:`torchmin.minimize()`.

    """
    def __init__(self,
                 params,
                 method='bfgs',
                 **minimize_kwargs):
        assert isinstance(method, str)
        method_ = method.lower()

        self._hessp = self._hess = False
        if method_ in ['bfgs', 'l-bfgs', 'cg']:
            pass
        elif method_ in ['newton-cg', 'trust-ncg', 'trust-krylov']:
            self._hessp = True
        elif method_ in ['newton-exact', 'dogleg', 'trust-exact']:
            self._hess = True
        else:
            raise ValueError('Unknown method {}'.format(method))

        defaults = dict(method=method_, **minimize_kwargs)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Minimizer doesn't support per-parameter options")

        self._nfev = [0]
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._closure = None
        self._result = None

    @property
    def nfev(self):
        return self._nfev[0]

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_param(self):
        params = []
        for p in self._params:
            if p.data.is_sparse:
                p = p.data.to_dense().view(-1)
            else:
                p = p.data.view(-1)
            params.append(p)
        return torch.cat(params)

    def _gather_flat_grad(self):
        grads = []
        for p in self._params:
            if p.grad is None:
                g = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                g = p.grad.to_dense().view(-1)
            else:
                g = p.grad.view(-1)
            grads.append(g)
        return torch.cat(grads)

    def _set_flat_param(self, value):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.copy_(value[offset:offset+numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    def closure(self, x):
        from torchmin.function import sf_value

        assert self._closure is not None
        self._set_flat_param(x)
        with torch.enable_grad():
            f = self._closure()
            f.backward(create_graph=self._hessp or self._hess)
            grad = self._gather_flat_grad()

        grad_out = grad.detach().clone()
        hessp = None
        hess = None
        if self._hessp or self._hess:
            grad_accum = grad.detach().clone()
            def hvp(v):
                assert v.shape == grad.shape
                grad.backward(gradient=v, retain_graph=True)
                output = self._gather_flat_grad().detach() - grad_accum
                grad_accum.add_(output)
                return output

            numel = self._numel()
            if self._hessp:
                hessp = LinearOperator(hvp, shape=(numel, numel),
                                       dtype=grad.dtype, device=grad.device)
            if self._hess:
                eye = torch.eye(numel, dtype=grad.dtype, device=grad.device)
                hess = torch.zeros(numel, numel, dtype=grad.dtype, device=grad.device)
                for i in range(numel):
                    hess[i] = hvp(eye[i])

        return sf_value(f=f.detach(), grad=grad_out.detach(), hessp=hessp, hess=hess)

    def dir_evaluate(self, x, t, d):
        from torchmin.function import de_value

        self._set_flat_param(x + d.mul(t))
        with torch.enable_grad():
            f = self._closure()
        f.backward()
        grad = self._gather_flat_grad()
        self._set_flat_param(x)

        return de_value(f=float(f), grad=grad)

    @torch.no_grad()
    def step(self, closure):
        """Perform an optimization step.

        The function "closure" should have a slightly different
        form vs. the PyTorch standard: namely, it should not include any
        `backward()` calls. Backward steps will be performed internally
        by the optimizer.

        >>> def closure():
        >>>    optimizer.zero_grad()
        >>>    output = model(input)
        >>>    loss = loss_fn(output, target)
        >>>    # loss.backward() <-- skip this step!
        >>>    return loss

        Parameters
        ----------
        closure : callable
            A function that re-evaluates the model and returns the loss.

        """
        from torchmin.minimize import minimize

        # sanity check
        assert len(self.param_groups) == 1

        # overwrite closure
        closure_ = closure
        def closure():
            self._nfev[0] += 1
            return closure_()
        self._closure = closure

        # get initial value
        x0 = self._gather_flat_param()

        # perform parameter update
        kwargs = {k:v for k,v in self.param_groups[0].items() if k != 'params'}
        self._result = minimize(self, x0, **kwargs)

        # set final value
        self._set_flat_param(self._result.x)

        return self._result.fun
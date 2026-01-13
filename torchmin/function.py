from typing import List, Optional
from torch import Tensor
from collections import namedtuple
import torch
import torch.autograd as autograd

from .optim.minimizer import Minimizer

__all__ = ['ScalarFunction', 'VectorFunction']



# scalar function result (value)
sf_value = namedtuple('sf_value', ['f', 'grad', 'hessp', 'hess'])

# directional evaluate result
de_value = namedtuple('de_value', ['f', 'grad'])

# vector function result (value)
vf_value = namedtuple('vf_value', ['f', 'jacp', 'jac'])


#@torch.jit.script
class JacobianLinearOperator(object):
    def __init__(self, x, func, symmetric: bool = False):
        # Compute current function value and vjp callable
        f, vjp_func = torch.func.vjp(func, x)

        # core properties
        self.x = x
        self.f = f
        self.func = func
        self.vjp_func = vjp_func
        self.symmetric = symmetric

        # tensor-like properties
        self.shape = (f.numel(), x.numel())
        self.dtype = x.dtype
        self.device = x.device

    def mv(self, v: Tensor) -> Tensor:
        if self.symmetric:
            return self.rmv(v)
        return torch.func.jvp(self.func, (self.x,), (v,))[1]

    def rmv(self, v: Tensor) -> Tensor:
        return self.vjp_func(v)[0]


class ScalarFunction(object):
    """Scalar-valued objective function with autograd backend.

    This class provides a general-purpose objective wrapper which will
    compute first- and second-order derivatives via autograd as specified
    by the parameters of __init__.
    """
    def __new__(cls, fun, x_shape, hessp=False, hess=False, twice_diffable=True):
        if isinstance(fun, Minimizer):
            assert fun._hessp == hessp
            assert fun._hess == hess
            return fun
        return super(ScalarFunction, cls).__new__(cls)

    def __init__(self, fun, x_shape, hessp=False, hess=False, twice_diffable=True):
        self._fun = fun
        self._x_shape = x_shape
        self._hessp = hessp
        self._hess = hess
        self._I = None
        self._twice_diffable = twice_diffable
        self.nfev = 0

    def fun(self, x):
        if x.shape != self._x_shape:
            x = x.view(self._x_shape)
        f = self._fun(x)
        if f.numel() != 1:
            raise RuntimeError('ScalarFunction was supplied a function '
                               'that does not return scalar outputs.')
        self.nfev += 1

        return f

    def closure(self, x):
        """Evaluate the function, gradient, and hessian/hessian-product

        This method represents the core function call. It is used for
        computing newton/quasi newton directions, etc.
        """
        x = x.detach().requires_grad_(True)

        f = grad = hessp = hess = None

        with torch.enable_grad():
            f = self.fun(x)
            if not self._hessp:
                grad = autograd.grad(f, x)[0]

        jac_fn = None
        if self._hessp or self._hess:
            # jac_fn = torch.func.jacrev(self.fun)
            jac_fn = torch.func.grad(self.fun)
        if self._hessp:
            hessp = JacobianLinearOperator(x, jac_fn, symmetric=self._twice_diffable)
            grad = hessp.f
        if self._hess:
            #hess = torch.func.hessian(self.fun)(x)
            hess = torch.func.jacfwd(jac_fn)(x)

        return sf_value(f=f.detach(), grad=grad.detach(), hessp=hessp, hess=hess)

    def dir_evaluate(self, x, t, d):
        """Evaluate a direction and step size.

        We define a separate "directional evaluate" function to be used
        for strong-wolfe line search. Only the function value and gradient
        are needed for this use case, so we avoid computational overhead.
        """
        x = x + d.mul(t)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
        grad = autograd.grad(f, x)[0]

        return de_value(f=float(f), grad=grad)


class VectorFunction(object):
    """Vector-valued objective function with autograd backend."""
    def __init__(self, fun, x_shape, jacp=False, jac=False):
        self._fun = fun
        self._x_shape = x_shape
        self._jacp = jacp
        self._jac = jac
        self._I = None
        self.nfev = 0

    def fun(self, x):
        if x.shape != self._x_shape:
            x = x.view(self._x_shape)
        f = self._fun(x)
        if f.dim() == 0:
            raise RuntimeError('VectorFunction expected vector outputs but '
                               'received a scalar.')
        elif f.dim() > 1:
            f = f.view(-1)
        self.nfev += 1

        return f

    def closure(self, x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
        jacp = None
        jac = None
        if self._jacp:
            jacp = JacobianLinearOperator(x, self.fun)
        if self._jac:
            jac = torch.func.jacfwd(self.fun)(x)

        return vf_value(f=f.detach(), jacp=jacp, jac=jac)
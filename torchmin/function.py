from typing import List, Optional
from torch import Tensor
from collections import namedtuple
import torch
import torch.autograd as autograd
from torch._vmap_internals import _vmap

from .optim.minimizer import Minimizer

__all__ = ['ScalarFunction', 'VectorFunction']



# scalar function result (value)
sf_value = namedtuple('sf_value', ['f', 'grad', 'hessp', 'hess'])

# directional evaluate result
de_value = namedtuple('de_value', ['f', 'grad'])

# vector function result (value)
vf_value = namedtuple('vf_value', ['f', 'jacp', 'jac'])


@torch.jit.script
class JacobianLinearOperator(object):
    def __init__(self,
                 x: Tensor,
                 f: Tensor,
                 gf: Optional[Tensor] = None,
                 gx: Optional[Tensor] = None,
                 symmetric: bool = False) -> None:
        self.x = x
        self.f = f
        self.gf = gf
        self.gx = gx
        self.symmetric = symmetric
        # tensor-like properties
        self.shape = (x.numel(), x.numel())
        self.dtype = x.dtype
        self.device = x.device

    def mv(self, v: Tensor) -> Tensor:
        if self.symmetric:
            return self.rmv(v)
        assert v.shape == self.x.shape
        gx, gf = self.gx, self.gf
        assert (gx is not None) and (gf is not None)
        outputs: List[Tensor] = [gx]
        inputs: List[Tensor] = [gf]
        grad_outputs: List[Optional[Tensor]] = [v]
        jvp = autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)[0]
        if jvp is None:
            raise Exception
        return jvp

    def rmv(self, v: Tensor) -> Tensor:
        assert v.shape == self.f.shape
        outputs: List[Tensor] = [self.f]
        inputs: List[Tensor] = [self.x]
        grad_outputs: List[Optional[Tensor]] = [v]
        vjp = autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)[0]
        if vjp is None:
            raise Exception
        return vjp


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
        with torch.enable_grad():
            f = self.fun(x)
            grad = autograd.grad(f, x, create_graph=self._hessp or self._hess)[0]
        if (self._hessp or self._hess) and grad.grad_fn is None:
            raise RuntimeError('A 2nd-order derivative was requested but '
                               'the objective is not twice-differentiable.')
        hessp = None
        hess = None
        if self._hessp:
            hessp = JacobianLinearOperator(x, grad, symmetric=self._twice_diffable)
        if self._hess:
            if self._I is None:
                self._I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            hvp = lambda v: autograd.grad(grad, x, v, retain_graph=True)[0]
            hess = _vmap(hvp)(self._I)

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
            jacp = JacobianLinearOperator(x, f)
        if self._jac:
            if self._I is None:
                self._I = torch.eye(f.numel(), dtype=x.dtype, device=x.device)
            vjp = lambda v: autograd.grad(f, x, v, retain_graph=True)[0]
            jac = _vmap(vjp)(self._I)

        return vf_value(f=f.detach(), jacp=jacp, jac=jac)
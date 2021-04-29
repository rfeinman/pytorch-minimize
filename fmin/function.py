from typing import List, Optional
from torch import Tensor
from collections import namedtuple
import torch
import torch.autograd as autograd
from torch._vmap_internals import _vmap

__all__ = ['ScalarFunction', 'DirectionalEvaluate']

sf_value = namedtuple('sf_value', ['f', 'grad', 'hessp', 'hess'])
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
    def __init__(self, fun, x_shape, hessp=False, hess=False,
                 twice_diffable=True):
        self.__fun = fun
        self._x_shape = x_shape
        self._hessp = hessp
        self._hess = hess
        self._I = None
        self._twice_diffable = twice_diffable
        self.nfev = 0

    def _fun(self, x):
        if x.shape != self._x_shape:
            x = x.view(self._x_shape)
        return self.__fun(x)

    def __call__(self, x):
        self.nfev += 1
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self._fun(x)
            if f.numel() != 1:
                raise RuntimeError('ScalarFunction was supplied a function '
                                   'that does not return scalar outputs.')
            grad = autograd.grad(f, x, create_graph=self._hessp or self._hess)[0]
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


class DirectionalEvaluate(ScalarFunction):
    def __init__(self, fun, x_shape):
        super().__init__(fun, x_shape)

    def __call__(self, x, t, d):
        x = x + t * d
        f, grad, _, _ = super().__call__(x)
        return float(f), grad


class VectorFunction(object):
    def __init__(self, fun, x_shape=None, jacp=False, jac=False):
        if x_shape is not None:
            fun_ = fun
            fun = lambda x: fun_(x.view(x_shape)).view(-1)
        self._fun = fun
        self._jacp = jacp
        self._jac = jac
        self._I = None
        self.nfev = 0

    def __call__(self, x):
        self.nfev += 1
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self._fun(x)
            if f.dim() == 0:
                raise RuntimeError('VectorFunction expected vector outputs but '
                                   'received a scalar.')
        jacp = None
        jac = None
        if self._jacp:
            jacp = JacobianLinearOperator(x, f)
        if self._jac:
            if self._I is None:
                self._I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            jvp = lambda v: autograd.grad(f, x, v, retain_graph=True)[0]
            jac = _vmap(jvp)(self._I)

        return vf_value(f=f.detach(), jacp=jacp, jac=jac)
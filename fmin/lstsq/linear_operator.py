import torch
import torch.autograd as autograd
from torch._vmap_internals import _vmap


def jacobian_dense(fun, x, vectorize=True):
    x = x.detach().requires_grad_(True)
    return autograd.functional.jacobian(fun, x, vectorize=vectorize)


def jacobian_linop(fun, x, return_f=False):
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        f = fun(x)

    # vector-jacobian product
    def vjp(v):
        v = v.view_as(f)
        vjp, = autograd.grad(f, x, v, retain_graph=True)
        return vjp.view(-1)

    # jacobian-vector product
    gf = torch.zeros_like(f, requires_grad=True)
    with torch.enable_grad():
        gx, = autograd.grad(f, x, gf, create_graph=True)
    def jvp(v):
        v = v.view_as(x)
        jvp, = autograd.grad(gx, gf, v, retain_graph=True)
        return jvp.view(-1)

    jac = TorchLinearOperator((f.numel(), x.numel()), matvec=jvp, rmatvec=vjp)

    if return_f:
        return jac, f.detach()
    return jac


class TorchLinearOperator(object):
    """Linear operator defined in terms of user-specified operations."""
    def __init__(self, shape, matvec, rmatvec):
        self.shape = shape
        self._matvec = matvec
        self._rmatvec = rmatvec

    def matvec(self, x):
        return self._matvec(x)

    def rmatvec(self, x):
        return self._rmatvec(x)

    def matmat(self, X):
        try:
            return _vmap(self.matvec)(X.T).T
        except:
            return torch.hstack([self.matvec(col).view(-1,1) for col in X.T])

    def transpose(self):
        new_shape = (self.shape[1], self.shape[0])
        return type(self)(new_shape, self._rmatvec, self._matvec)

    mv = matvec
    rmv = rmatvec
    matmul = matmat
    t = transpose
    T = property(transpose)


def aslinearoperator(A):
    if isinstance(A, TorchLinearOperator):
        return A
    elif isinstance(A, torch.Tensor):
        assert A.dim() == 2
        return TorchLinearOperator(A.shape, matvec=A.mv, rmatvec=A.T.mv)
    else:
        raise ValueError('Input must be either a Tensor or TorchLinearOperator')
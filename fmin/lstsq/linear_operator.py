import torch
import torch.autograd as autograd


def jacobian_linop(fun, x):
    x = x.detach().requires_grad_(True)
    f = fun(x)

    # vector-jacobian product
    def vjp(v):
        v = v.view_as(f)
        vjp, = autograd.grad(f, x, v, retain_graph=True)
        return vjp.view(-1)

    # jacobian-vector product
    gf = torch.zeros_like(f, requires_grad=True)
    gx, = autograd.grad(f, x, gf, create_graph=True)
    def jvp(v):
        v = v.view_as(x)
        jvp, = autograd.grad(gx, gf, v, retain_graph=True)
        return jvp.view(-1)

    jac = TorchLinearOperator((f.numel(), x.numel()), matvec=jvp, rmatvec=vjp)

    return jac


class TorchLinearOperator(object):
    """Linear operator defined in terms of user-specified operations."""
    def __init__(self, shape, matvec, rmatvec=None):
        self.shape = shape
        self._matvec = matvec
        self._rmatvec = rmatvec

    def matvec(self, x):
        return self._matvec(x)

    def rmatvec(self, x):
        if self._rmatvec is None:
            raise NotImplementedError("rmatvec is not defined")
        return self._rmatvec(x)

    def matmat(self, X):
        return torch.hstack([self.matvec(col).view(-1,1) for col in X.T])

    mv = matvec
    rmv = rmatvec
    matmul = matmat
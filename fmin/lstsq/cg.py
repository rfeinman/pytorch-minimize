import torch

from .linear_operator import aslinearoperator, TorchLinearOperator


def cg(A, b, x0=None, max_iter=None, tol=1e-5):
    if max_iter is None:
        max_iter = 20 * b.numel()
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A.mv(x)
    p = r.clone()
    rs = r.dot(r)
    for n_iter in range(1, max_iter+1):
        Ap = A.mv(p)
        alpha = rs / p.dot(Ap)
        x.add_(p, alpha=alpha)
        r.sub_(Ap, alpha=alpha)
        rs_new = r.dot(r)
        p.mul_(rs_new / rs).add_(r)
        if n_iter % 10 == 0:
            r_norm = rs.sqrt()
            if r_norm < tol:
                break
        rs = rs_new

    return x


def cgls(A, b, alpha=0., **kwargs):
    A = aslinearoperator(A)
    m, n = A.shape
    Atb = A.rmv(b)
    AtA = TorchLinearOperator(shape=(n,n),
                              matvec=lambda x: A.rmv(A.mv(x)) + alpha * x,
                              rmatvec=None)
    return cg(AtA, Atb, **kwargs)
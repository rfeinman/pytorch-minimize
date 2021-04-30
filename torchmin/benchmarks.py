import torch

__all__ = ['rosen', 'rosen_der', 'rosen_hess', 'rosen_hess_prod']


# =============================
#     Rosenbrock function
# =============================


def rosen(x, reduce=True):
    val = 100. * (x[...,1:] - x[...,:-1]**2)**2 + (1 - x[...,:-1])**2
    if reduce:
        return val.sum()
    else:
        # don't reduce batch dimensions
        return val.sum(-1)


def rosen_der(x):
    xm = x[..., 1:-1]
    xm_m1 = x[..., :-2]
    xm_p1 = x[..., 2:]
    der = torch.zeros_like(x)
    der[..., 1:-1] = (200 * (xm - xm_m1**2) -
                 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    der[..., 0] = -400 * x[..., 0] * (x[..., 1] - x[..., 0]**2) - 2 * (1 - x[..., 0])
    der[..., -1] = 200 * (x[..., -1] - x[..., -2]**2)
    return der


def rosen_hess(x):
    H = torch.diag_embed(-400*x[..., :-1], 1) - \
        torch.diag_embed(400*x[..., :-1], -1)
    diagonal = torch.zeros_like(x)
    diagonal[..., 0] = 1200*x[..., 0].square() - 400*x[..., 1] + 2
    diagonal[..., -1] = 200
    diagonal[..., 1:-1] = 202 + 1200*x[..., 1:-1].square() - 400*x[..., 2:]
    H.diagonal(dim1=-2, dim2=-1).add_(diagonal)
    return H


def rosen_hess_prod(x, p):
    Hp = torch.zeros_like(x)
    Hp[..., 0] = (1200 * x[..., 0]**2 - 400 * x[..., 1] + 2) * p[..., 0] - \
                 400 * x[..., 0] * p[..., 1]
    Hp[..., 1:-1] = (-400 * x[..., :-2] * p[..., :-2] +
                     (202 + 1200 * x[..., 1:-1]**2 - 400 * x[..., 2:]) * p[..., 1:-1] -
                     400 * x[..., 1:-1] * p[..., 2:])
    Hp[..., -1] = -400 * x[..., -2] * p[..., -2] + 200*p[..., -1]
    return Hp
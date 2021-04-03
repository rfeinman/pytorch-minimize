import torch


_status_messages = {
    0: 'Absolute tolerance reached',
    1: 'Relative tolerance reached',
    2: 'Curvature has converged',
    3: 'Curvature is negative',
    4: 'Maximum iterations reached'
}


def conjgrad(b, Adot, dot=None, max_iter=None, tol=1e-10, rtol=1e-1, disp=0,
             return_info=False):
    if max_iter is None:
        max_iter = 20 * b.numel()
    if dot is None:
        dot = lambda u,v: u.mul(v).sum()
    disp = int(disp)
    b_norm = b.norm(p=1)
    termcond = rtol * b_norm * b_norm.sqrt().clamp(0, 0.5)
    eps = torch.finfo(b.dtype).eps

    # initialize state
    x = torch.zeros_like(b)
    r = b.neg()
    p = b
    rs = dot(r, r)
    n_iter = 0

    # termination func
    def terminate(status):
        if disp:
            print('ConjGrad: ' + _status_messages[status])
        if return_info:
            return x, n_iter, status
        return x

    # iterate
    while n_iter < max_iter:
        if r.norm(p=2) < tol:
            return terminate(0)
        if r.norm(p=1) <= termcond:
            return terminate(1)
        Ap = Adot(p)
        curv = dot(p, Ap)
        curv_sum = curv.sum()
        if 0 <= curv_sum <= 3 * eps:
            return terminate(2)
        elif curv_sum < 0:
            if n_iter == 0:
                # fall back to steepest descent direction
                x = - rs / curv * b
            return terminate(3)
        alpha = rs / curv
        x = x + alpha * p
        r = r + alpha * Ap
        rs_new = dot(r, r)
        p = - r + (rs_new / rs) * p
        rs = rs_new
        n_iter += 1
        if disp > 1:
            print('iter: %i - rs: %0.4f' % (n_iter, rs.sum()))

    return terminate(4)
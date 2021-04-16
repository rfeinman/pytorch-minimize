import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator

from .linear_operator import TorchLinearOperator

EPS = torch.finfo(float).eps


def in_bounds(x, lb, ub):
    """Check if a point lies within bounds."""
    return torch.all((x >= lb) & (x <= ub))

def find_active_constraints(x, lb, ub, rtol=1e-10):
    """Determine which constraints are active in a given point.
    The threshold is computed using `rtol` and the absolute value of the
    closest bound.
    Returns
    -------
    active : ndarray of int with shape of x
        Each component shows whether the corresponding constraint is active:
             *  0 - a constraint is not active.
             * -1 - a lower bound is active.
             *  1 - a upper bound is active.
    """
    active = torch.zeros_like(x, dtype=torch.long)

    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active

    lower_dist = x - lb
    upper_dist = ub - x
    lower_threshold = rtol * lb.abs().clamp(1, None)
    upper_threshold = rtol * ub.abs().clamp(1, None)

    lower_active = (lb.isfinite() &
                    (lower_dist <= torch.minimum(upper_dist, lower_threshold)))
    active[lower_active] = -1

    upper_active = (ub.isfinite() &
                    (upper_dist <= torch.minimum(lower_dist, upper_threshold)))
    active[upper_active] = 1

    return active

def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    """Shift a point to the interior of a feasible region.
    Each element of the returned vector is at least at a relative distance
    `rstep` from the closest bound. If ``rstep=0`` then `np.nextafter` is used.
    """
    x_new = x.clone()

    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = torch.eq(active, -1)
    upper_mask = torch.eq(active, 1)

    if rstep == 0:
        torch.nextafter(lb[lower_mask], ub[lower_mask], out=x_new[lower_mask])
        torch.nextafter(ub[upper_mask], lb[upper_mask], out=x_new[upper_mask])
    else:
        x_new[lower_mask] = lb[lower_mask].add(lb[lower_mask].abs().clamp(1,None), alpha=rstep)
        x_new[upper_mask] = ub[upper_mask].sub(ub[upper_mask].abs().clamp(1,None), alpha=rstep)

    tight_bounds = (x_new < lb) | (x_new > ub)
    x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])

    return x_new


def right_multiplied_operator(J, d):
    """Return J diag(d) as LinearOperator."""
    if isinstance(J, LinearOperator):
        if torch.is_tensor(d):
            d = d.data.cpu().numpy()
        return LinearOperator(J.shape,
                              matvec=lambda x: J.matvec(np.ravel(x) * d),
                              matmat=lambda X: J.matmat(X * d[:, np.newaxis]),
                              rmatvec=lambda x: d * J.rmatvec(x))
    elif isinstance(J, TorchLinearOperator):
        return TorchLinearOperator(J.shape,
                                   matvec=lambda x: J.matvec(x.view(-1) * d),
                                   rmatvec=lambda x: d * J.rmatvec(x))
    else:
        raise ValueError('Expected J to be a LinearOperator or '
                         'TorchLinearOperator but found {}'.format(type(J)))

def build_quadratic_1d(J, g, s, diag=None, s0=None):
    """Parameterize a multivariate quadratic function along a line.

    The resulting univariate quadratic function is given as follows:
    ::
        f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +
               g.T * (s0 + s*t)
    """
    v = J.mv(s)
    a = v.dot(v)
    if diag is not None:
        a += s.dot(s * diag)
    a *= 0.5

    b = g.dot(s)

    if s0 is not None:
        u = J.mv(s0)
        b += u.dot(v)
        c = 0.5 * u.dot(u) + g.dot(s0)
        if diag is not None:
            b += s.dot(s0 * diag)
            c += 0.5 * s0.dot(s0 * diag)
        return a, b, c
    else:
        return a, b


def minimize_quadratic_1d(a, b, lb, ub, c=0):
    """Minimize a 1-D quadratic function subject to bounds.

    The free term `c` is 0 by default. Bounds must be finite.
    """
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = a.new_tensor(t)
    y = t * (a * t + b) + c
    min_index = torch.argmin(y)
    return t[min_index], y[min_index]


def evaluate_quadratic(J, g, s, diag=None):
    """Compute values of a quadratic function arising in least squares.
    The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.
    """
    if s.dim() == 1:
        Js = J.mv(s)
        q = Js.dot(Js)
        if diag is not None:
            q += s.dot(s * diag)
    else:
        Js = J.matmul(s.T)
        q = Js.square().sum(0)
        if diag is not None:
            q += (diag * s.square()).sum(1)

    l = s.matmul(g)

    return 0.5 * q + l


def solve_trust_region_2d(B, g, Delta):
    """Solve a general trust-region problem in 2 dimensions.
    The problem is reformulated as a 4th order algebraic equation,
    the solution of which is found by numpy.roots.
    """
    try:
        L = torch.cholesky(B)
        p = - torch.cholesky_solve(g.unsqueeze(1), L).squeeze(1)
        if p.dot(p) <= Delta**2:
            return p, True
    except RuntimeError as exc:
        if not 'cholesky' in exc.args[0]:
            raise
        pass

    a = B[0, 0] * Delta**2
    b = B[0, 1] * Delta**2
    c = B[1, 1] * Delta**2
    d = g[0] * Delta
    f = g[1] * Delta

    coeffs = B.new_tensor([-b + d,
                           2 * (a - c + f),
                           6 * b,
                           2 * (-a + c + f),
                           -b - d])

    # TODO: pytorch implementation of np.roots?
    t = np.roots(coeffs.data.cpu().numpy())  # Can handle leading zeros.
    t = torch.tensor(t, device=B.device)
    t = torch.real(t[torch.isreal(t)])

    p = Delta * torch.vstack((2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)))
    value = 0.5 * torch.sum(p * B.matmul(p), 0) + g.matmul(p)
    i = torch.argmin(value)
    p = p[:, i]

    return p, False


def update_tr_radius(Delta, actual_reduction, predicted_reduction,
                     step_norm, bound_hit):
    """Update the radius of a trust region based on the cost reduction.
    """
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0

    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0

    return Delta, ratio


def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    """Check termination condition for nonlinear least squares."""
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None
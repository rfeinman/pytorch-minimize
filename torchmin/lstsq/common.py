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


def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None,
                           rtol=0.01, max_iter=10):
    """Solve a trust-region problem arising in least-squares minimization.
    This function implements a method described by J. J. More [1]_ and used
    in MINPACK, but it relies on a single SVD of Jacobian instead of series
    of Cholesky decompositions. Before running this function, compute:
    ``U, s, VT = svd(J, full_matrices=False)``.
    """
    def phi_and_derivative(alpha, suf, s, Delta):
        """Function of which to find zero.
        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `Delta`". Refer to [1]_.
        """
        denom = s.pow(2) + alpha
        p_norm = (suf / denom).norm()
        phi = p_norm - Delta
        phi_prime = -(suf.pow(2) / denom.pow(3)).sum() / p_norm
        return phi, phi_prime

    def set_alpha(alpha_lower, alpha_upper):
        new_alpha = (alpha_lower * alpha_upper).sqrt()
        return new_alpha.clamp_(0.001 * alpha_upper, None)

    suf = s * uf

    # Check if J has full rank and try Gauss-Newton step.
    eps = torch.finfo(s.dtype).eps
    full_rank = m >= n and s[-1] > eps * m * s[0]

    if full_rank:
        p = -V.mv(uf / s)
        if p.norm() <= Delta:
            return p, 0.0, 0
        phi, phi_prime = phi_and_derivative(0., suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = s.new_tensor(0.)

    alpha_upper = suf.norm() / Delta

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        alpha = set_alpha(alpha_lower, alpha_upper)
    else:
        alpha = initial_alpha.clone()

    for it in range(max_iter):
        # if alpha is outside of bounds, set new value (5.5)(a)
        alpha = torch.where((alpha < alpha_lower) | (alpha > alpha_upper),
                            set_alpha(alpha_lower, alpha_upper),
                            alpha)

        # compute new phi and phi' (5.5)(b)
        phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)

        # if phi is negative, update our upper bound  (5.5)(b)
        alpha_upper = torch.where(phi < 0, alpha, alpha_upper)

        # update lower bound  (5.5)(b)
        ratio = phi / phi_prime
        alpha_lower.clamp_(alpha-ratio, None)

        # compute new alpha (5.5)(c)
        alpha.addcdiv_((phi + Delta) * ratio, Delta, value=-1)

        if phi.abs() < rtol * Delta:
            break

    p = -V.mv(suf / (s.pow(2) + alpha))

    # Make the norm of p equal to Delta, p is changed only slightly during
    # this. It is done to prevent p lie outside the trust region (which can
    # cause problems later).
    p.mul_(Delta / p.norm())

    return p, alpha, it + 1


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
        L = torch.linalg.cholesky(B)
        p = - torch.cholesky_solve(g.unsqueeze(1), L).squeeze(1)
        if p.dot(p) <= Delta**2:
            return p, True
    except RuntimeError as exc:
        if not 'cholesky' in exc.args[0]:
            raise

    # move things to numpy
    device = B.device
    dtype = B.dtype
    B = B.data.cpu().numpy()
    g = g.data.cpu().numpy()
    Delta = float(Delta)

    a = B[0, 0] * Delta**2
    b = B[0, 1] * Delta**2
    c = B[1, 1] * Delta**2
    d = g[0] * Delta
    f = g[1] * Delta

    coeffs = np.array([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.roots(coeffs)  # Can handle leading zeros.
    t = np.real(t[np.isreal(t)])

    p = Delta * np.vstack((2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)))
    value = 0.5 * np.sum(p * B.dot(p), axis=0) + np.dot(g, p)
    p = p[:, np.argmin(value)]

    # convert back to torch
    p = torch.tensor(p, device=device, dtype=dtype)

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
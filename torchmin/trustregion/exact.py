"""
Nearly exact trust-region optimization subproblem.

Code ported from SciPy to PyTorch

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.
"""
from typing import Tuple
from torch import Tensor
import torch
from torch.linalg import norm
from scipy.linalg import get_lapack_funcs

from .base import _minimize_trust_region, BaseQuadraticSubproblem


def _minimize_trust_exact(fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    a nearly exact trust-region algorithm.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    initial_tr_radius : float
        Initial trust-region radius.
    max_tr_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    Notes
    -----
    This trust-region solver was based on [1]_, [2]_ and [3]_,
    which implement similar algorithms. The algorithm is basically
    that of [1]_ but ideas from [2]_ and [3]_ were also used.

    References
    ----------
    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
           Siam, pp. 169-200, 2000.
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
           pp. 553-572, 1983.

    """
    return _minimize_trust_region(fun, x0,
                                  subproblem=IterativeSubproblem,
                                  **trust_region_options)


def solve_triangular(A, b, **kwargs):
    return torch.triangular_solve(b.unsqueeze(1), A, **kwargs)[0].squeeze(1)


def solve_cholesky(A, b, **kwargs):
    return torch.cholesky_solve(b.unsqueeze(1), A, **kwargs).squeeze(1)


@torch.jit.script
def estimate_smallest_singular_value(U) -> Tuple[Tensor, Tensor]:
    """Given upper triangular matrix ``U`` estimate the smallest singular
    value and the correspondent right singular vector in O(n**2) operations.

    A vector `e` with components selected from {+1, -1}
    is selected so that the solution `w` to the system
    `U.T w = e` is as large as possible. Implementation
    based on algorithm 3.5.1, p. 142, from reference [1]_
    adapted for lower triangular matrix.

    References
    ----------
    .. [1] G.H. Golub, C.F. Van Loan. "Matrix computations".
           Forth Edition. JHU press. pp. 140-142.
    """

    U = torch.atleast_2d(U)
    UT = U.T
    m, n = U.shape
    if m != n:
        raise ValueError("A square triangular matrix should be provided.")

    p = torch.zeros(n, dtype=U.dtype, device=U.device)
    w = torch.empty(n, dtype=U.dtype, device=U.device)

    for k in range(n):
        wp = (1-p[k]) / UT[k, k]
        wm = (-1-p[k]) / UT[k, k]
        pp = p[k+1:] + UT[k+1:, k] * wp
        pm = p[k+1:] + UT[k+1:, k] * wm

        if wp.abs() + norm(pp, 1) >= wm.abs() + norm(pm, 1):
            w[k] = wp
            p[k+1:] = pp
        else:
            w[k] = wm
            p[k+1:] = pm

    # The system `U v = w` is solved using backward substitution.
    v = torch.triangular_solve(w.view(-1,1), U)[0].view(-1)
    v_norm = norm(v)

    s_min = norm(w) / v_norm  # Smallest singular value
    z_min = v / v_norm        # Associated vector

    return s_min, z_min


def gershgorin_bounds(H):
    """
    Given a square matrix ``H`` compute upper
    and lower bounds for its eigenvalues (Gregoshgorin Bounds).
    """
    H_diag = torch.diag(H)
    H_diag_abs = H_diag.abs()
    H_row_sums = H.abs().sum(dim=1)
    lb = torch.min(H_diag + H_diag_abs - H_row_sums)
    ub = torch.max(H_diag - H_diag_abs + H_row_sums)

    return lb, ub


def singular_leading_submatrix(A, U, k):
    """
    Compute term that makes the leading ``k`` by ``k``
    submatrix from ``A`` singular.
    """
    u = U[:k-1, k-1]

    # Compute delta
    delta = u.dot(u) - A[k-1, k-1]

    # Initialize v
    v = A.new_zeros(A.shape[0])
    v[k-1] = 1

    # Compute the remaining values of v by solving a triangular system.
    if k != 1:
        v[:k-1] = solve_triangular(U[:k-1, :k-1], -u)

    return delta, v


class IterativeSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by nearly exact iterative method."""

    # UPDATE_COEFF appears in reference [1]_
    # in formula 7.3.14 (p. 190) named as "theta".
    # As recommended there it value is fixed in 0.01.
    UPDATE_COEFF = 0.01
    hess_prod = False

    def __init__(self, x, fun, k_easy=0.1, k_hard=0.2):

        super().__init__(x, fun)

        # When the trust-region shrinks in two consecutive
        # calculations (``tr_radius < previous_tr_radius``)
        # the lower bound ``lambda_lb`` may be reused,
        # facilitating  the convergence. To indicate no
        # previous value is known at first ``previous_tr_radius``
        # is set to -1  and ``lambda_lb`` to None.
        self.previous_tr_radius = -1
        self.lambda_lb = None

        self.niter = 0
        self.EPS = torch.finfo(x.dtype).eps

        # ``k_easy`` and ``k_hard`` are parameters used
        # to determine the stop criteria to the iterative
        # subproblem solver. Take a look at pp. 194-197
        # from reference _[1] for a more detailed description.
        self.k_easy = k_easy
        self.k_hard = k_hard

        # Get Lapack function for cholesky decomposition.
        # NOTE: cholesky_ex requires pytorch >= 1.9.0
        if 'cholesky_ex' in dir(torch.linalg):
            self.torch_cholesky = True
        else:
            # if we don't have torch cholesky, use potrf from scipy
            self.cholesky, = get_lapack_funcs(('potrf',),
                                              (self.hess.cpu().numpy(),))
            self.torch_cholesky = False

        # Get info about Hessian
        self.dimension = len(self.hess)
        self.hess_gershgorin_lb, self.hess_gershgorin_ub = gershgorin_bounds(self.hess)
        self.hess_inf = norm(self.hess, float('inf'))
        self.hess_fro = norm(self.hess, 'fro')

        # A constant such that for vectors smaler than that
        # backward substituition is not reliable. It was stabilished
        # based on Golub, G. H., Van Loan, C. F. (2013).
        # "Matrix computations". Forth Edition. JHU press., p.165.
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf

    def _initial_values(self, tr_radius):
        """Given a trust radius, return a good initial guess for
        the damping factor, the lower bound and the upper bound.
        The values were chosen accordingly to the guidelines on
        section 7.3.8 (p. 192) from [1]_.
        """
        hess_norm = torch.min(self.hess_fro, self.hess_inf)

        # Upper bound for the damping factor
        lambda_ub = self.jac_mag / tr_radius + torch.min(-self.hess_gershgorin_lb, hess_norm)
        lambda_ub = torch.clamp(lambda_ub, min=0)

        # Lower bound for the damping factor
        lambda_lb = self.jac_mag / tr_radius - torch.min(self.hess_gershgorin_ub, hess_norm)
        lambda_lb = torch.max(lambda_lb, -self.hess.diagonal().min())
        lambda_lb = torch.clamp(lambda_lb, min=0)

        # Improve bounds with previous info
        if tr_radius < self.previous_tr_radius:
            lambda_lb = torch.max(self.lambda_lb, lambda_lb)

        # Initial guess for the damping factor
        if lambda_lb == 0:
            lambda_initial = lambda_lb.clone()
        else:
            lambda_initial = torch.max(
                torch.sqrt(lambda_lb * lambda_ub),
                lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

        return lambda_initial, lambda_lb, lambda_ub

    def solve(self, tr_radius):
        """Solve quadratic subproblem"""

        lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
        n = self.dimension
        hits_boundary = True
        already_factorized = False
        self.niter = 0

        while True:
            # Compute Cholesky factorization
            if already_factorized:
                already_factorized = False
            else:
                H = self.hess.clone()
                H.diagonal().add_(lambda_current)
                if self.torch_cholesky:
                    U, info = torch.linalg.cholesky_ex(H)
                    U = U.t().contiguous()
                else:
                    U, info = self.cholesky(H.cpu().numpy(),
                                            lower=False,
                                            overwrite_a=False,
                                            clean=True)
                    U = H.new_tensor(U)

            self.niter += 1

            # Check if factorization succeeded
            if info == 0 and self.jac_mag > self.CLOSE_TO_ZERO:
                # Successful factorization

                # Solve `U.T U p = s`
                p = solve_cholesky(U, -self.jac, upper=True)
                p_norm = norm(p)

                # Check for interior convergence
                if p_norm <= tr_radius and lambda_current == 0:
                    hits_boundary = False
                    break

                # Solve `U.T w = p`
                w = solve_triangular(U, p, transpose=True)
                w_norm = norm(w)

                # Compute Newton step accordingly to
                # formula (4.44) p.87 from ref [2]_.
                delta_lambda = (p_norm/w_norm)**2 * (p_norm-tr_radius)/tr_radius
                lambda_new = lambda_current + delta_lambda

                if p_norm < tr_radius:  # Inside boundary
                    s_min, z_min = estimate_smallest_singular_value(U)

                    ta, tb = self.get_boundaries_intersections(p, z_min, tr_radius)

                    # Choose `step_len` with the smallest magnitude.
                    # The reason for this choice is explained at
                    # ref [3]_, p. 6 (Immediately before the formula
                    # for `tau`).
                    step_len = min(ta, tb, key=torch.abs)

                    # Compute the quadratic term  (p.T*H*p)
                    quadratic_term = p.dot(H.mv(p))

                    # Check stop criteria
                    relative_error = ((step_len**2 * s_min**2) /
                                      (quadratic_term + lambda_current*tr_radius**2))
                    if relative_error <= self.k_hard:
                        p.add_(step_len * z_min)
                        break

                    # Update uncertanty bounds
                    lambda_ub = lambda_current
                    lambda_lb = torch.max(lambda_lb, lambda_current - s_min**2)

                    # Compute Cholesky factorization
                    H = self.hess.clone()
                    H.diagonal().add_(lambda_new)
                    if self.torch_cholesky:
                        _, info = torch.linalg.cholesky_ex(H)
                    else:
                        _, info = self.cholesky(H.cpu().numpy(),
                                                lower=False,
                                                overwrite_a=False,
                                                clean=True)

                    if info == 0:
                        lambda_current = lambda_new
                        already_factorized = True
                    else:
                        lambda_lb = torch.max(lambda_lb, lambda_new)
                        lambda_current = torch.max(
                            torch.sqrt(lambda_lb * lambda_ub),
                            lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

                else:  # Outside boundary
                    # Check stop criteria
                    relative_error = torch.abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break

                    # Update uncertanty bounds
                    lambda_lb = lambda_current

                    # Update damping factor
                    lambda_current = lambda_new

            elif info == 0 and self.jac_mag <= self.CLOSE_TO_ZERO:
                # jac_mag very close to zero

                # Check for interior convergence
                if lambda_current == 0:
                    p = self.jac.new_zeros(n)
                    hits_boundary = False
                    break

                s_min, z_min = estimate_smallest_singular_value(U)
                step_len = tr_radius

                # Check stop criteria
                if step_len**2 * s_min**2 <= self.k_hard * lambda_current * tr_radius**2:
                    p = step_len * z_min
                    break

                # Update uncertainty bounds and dampening factor
                lambda_ub = lambda_current
                lambda_lb = torch.max(lambda_lb, lambda_current - s_min**2)
                lambda_current = torch.max(
                    torch.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

            else:
                # Unsuccessful factorization

                delta, v = singular_leading_submatrix(H, U, info)
                v_norm = norm(v)

                lambda_lb = torch.max(lambda_lb, lambda_current + delta/v_norm**2)

                # Update damping factor
                lambda_current = torch.max(
                    torch.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

        self.lambda_lb = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius

        return p, hits_boundary
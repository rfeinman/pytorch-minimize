"""
TODO: this module is not yet complete. It is not ready for use.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal, get_lapack_funcs
import torch

from .base import _minimize_trust_region, BaseQuadraticSubproblem


def _minimize_trust_krylov(fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    the GLTR Krylov subspace trust-region algorithm.

    .. warning::
        This minimizer is in early stages and has not been rigorously
        tested. It may change in the near future.

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
    This trust-region solver is based on the GLTR algorithm as
    described in [1]_ and [2]_.

    References
    ----------
    .. [1] F. Lenders, C. Kirches, and A. Potschka, "trlib: A vector-free
           implementation of the GLTR method for...",
           arXiv:1611.04718.
    .. [2] N. Gould, S. Lucidi, M. Roma, P. Toint: “Solving the Trust-Region
           Subproblem using the Lanczos Method”,
           SIAM J. Optim., 9(2), 504–525, 1999.
    .. [3] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.

    """
    return _minimize_trust_region(fun, x0,
                                  subproblem=KrylovSubproblem,
                                  **trust_region_options)


class KrylovSubproblem(BaseQuadraticSubproblem):
    """The GLTR trust region sub-problem defined on an expanding
    Krylov subspace.

    Based on the implementation of GLTR described in [1]_.

    References
    ----------
    .. [1] F. Lenders, C. Kirches, and A. Potschka, "trlib: A vector-free
           implementation of the GLTR method for...",
           arXiv:1611.04718.
    .. [2] N. Gould, S. Lucidi, M. Roma, P. Toint: “Solving the Trust-Region
           Subproblem using the Lanczos Method”,
           SIAM J. Optim., 9(2), 504–525, 1999.
    .. [3] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    """
    hess_prod = True
    max_lanczos = None
    max_ms_iters = 500  # max iterations of the Moré-Sorensen loop

    def __init__(self, x, fun, k_easy=0.1, k_hard=0.2, tol=1e-5, ortho=True,
                 debug=False):
        super().__init__(x, fun)
        self.eps = torch.finfo(x.dtype).eps
        self.k_easy = k_easy
        self.k_hard = k_hard
        self.tol = tol
        self.ortho = ortho
        self._debug = debug

    def tridiag_subproblem(self, Ta, Tb, tr_radius):
        """Solve the GLTR tridiagonal subproblem.

        Based on Algorithm 5.2 of [2]_. We factorize as follows:

        .. math::
            T + lambd * I = LDL^T

        Where `D` is diagonal and `L` unit (lower) bi-diagonal.
        """
        device, dtype = Ta.device, Ta.dtype

        # convert to numpy
        Ta = Ta.cpu().numpy()
        Tb = Tb.cpu().numpy()
        tr_radius = float(tr_radius)

        # right hand side
        rhs = np.zeros_like(Ta)
        rhs[0] = - float(self.jac_mag)

        # get LAPACK routines for factorizing and solving sym-PD tridiagonal
        ptsv, pttrs = get_lapack_funcs(('ptsv', 'pttrs'), (Ta, Tb, rhs))

        eig0 = None
        lambd_lb = 0.
        lambd = 0.
        for _ in range(self.max_ms_iters):
            lambd = max(lambd, lambd_lb)

            # factor T + lambd * I = LDL^T and solve LDL^T p = rhs
            d, e, p, info = ptsv(Ta + lambd, Tb, rhs)
            assert info >= 0  # sanity check
            if info > 0:
                assert eig0 is None  # sanity check; should only happen once
                # estimate smallest eigenvalue and continue
                eig0 = eigh_tridiagonal(
                    Ta, Tb, eigvals_only=True, select='i',
                    select_range=(0,0), lapack_driver='stebz').item()
                lambd_lb = max(1e-3 - eig0, 0)
                continue

            p_norm = np.linalg.norm(p)
            if p_norm < tr_radius:
                # TODO: add extra checks
                status = 0
                break
            elif abs(p_norm - tr_radius) / tr_radius <= self.k_easy:
                status = 1
                break

            # solve LDL^T q = p and compute <q, p>
            v, info = pttrs(d, e, p)
            q_norm2 = v.dot(p)

            # update lambd
            lambd += (p_norm**2 / q_norm2) * (p_norm - tr_radius) / tr_radius
        else:
            status = -1

        p = torch.tensor(p, device=device, dtype=dtype)

        return p, status, lambd

    def solve(self, tr_radius):
        g = self.jac
        gamma_0 = self.jac_mag
        n, = g.shape
        m = n if self.max_lanczos is None else min(n, self.max_lanczos)
        dtype = g.dtype
        device = g.device
        h_best = None
        error_best = float('inf')

        # Lanczos Q matrix buffer
        Q = torch.zeros(m, n, dtype=dtype, device=device)
        Q[0] = g / gamma_0

        # Lanczos T matrix buffers
        # a and b are the diagonal and off-diagonal entries of T, respectively
        a = torch.zeros(m, dtype=dtype, device=device)
        b = torch.zeros(m, dtype=dtype, device=device)

        # first lanczos iteration
        r = self.hessp(Q[0])
        torch.dot(Q[0], r, out=a[0])
        r.sub_(Q[0] * a[0])
        torch.linalg.norm(r, out=b[0])
        if b[0] < self.eps:
            raise RuntimeError('initial beta is zero.')

        # remaining iterations
        for i in range(1, m):
            torch.div(r, b[i-1], out=Q[i])
            r = self.hessp(Q[i])
            r.sub_(Q[i-1] * b[i-1])
            torch.dot(Q[i], r, out=a[i])
            r.sub_(Q[i] * a[i])
            if self.ortho:
                # Re-orthogonalize with Gram-Schmidt
                r.addmv_(Q[:i+1].T, Q[:i+1].mv(r), alpha=-1)
            torch.linalg.norm(r, out=b[i])
            if b[i] < self.eps:
                # This should never occur when self.ortho=True
                raise RuntimeError('reducible T matrix encountered.')

            # GLTR sub-problem
            h, status, lambd = self.tridiag_subproblem(a[:i+1], b[:i], tr_radius)

            if status >= 0:
                # convergence check; see Algorithm 1 of [1]_ and
                # Algorithm 5.1 of [2]_. Equivalent to the following:
                #     p = Q[:i+1].T.mv(h)
                #     error = torch.linalg.norm(self.hessp(p) + lambd * p + g)
                error = b[i] * h[-1].abs()
                if self._debug:
                    print('iter %3d - status: %d - lambd: %0.4e - error: %0.4e'
                          % (i+1, status, lambd, error))
                if error < error_best:
                    # we've found a new best
                    hits_boundary = status != 0
                    h_best = h
                    error_best = error
                    if error_best <= self.tol:
                        break

            elif self._debug:
                print('iter %3d - status: %d - lambd: %0.4e' %
                      (i+1, status, lambd))

        if h_best is None:
            # TODO: what should we do here?
            raise RuntimeError('gltr solution not found')

        # project h back to R^n
        p_best = Q[:i+1].T.mv(h_best)

        return p_best, hits_boundary

"""
TODO: this module is not yet complete. It is not ready for use.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal, get_lapack_funcs
import torch
try:
    # todo: port these functions from private "ptkit" library
    from ptkit.linalg import solveh_tridiag, eigh_tridiag
except:
    pass

from .base import BaseQuadraticSubproblem


class KrylovSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method.

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

    # extra variable defs
    lambd_0 = 1e-3
    max_lanczos = None
    max_ms_iters = 500  # max iterations of the Moré-Sorensen loop

    def __init__(self, x, fun, k_easy=0.1, k_hard=0.2, tol=1e-5, ortho=True, debug=False):
        super().__init__(x, fun)
        self.eps = torch.finfo(x.dtype).eps
        self.k_easy = k_easy
        self.k_hard = k_hard
        self.tol = tol
        self.ortho = ortho
        self.best_obj = float('inf')
        self._debug = debug

    def solve_krylov_mod2(self, Ta, Tb, gamma_0, tr_radius):
        """This version avoids factorization altogether and instead uses
        the very fast banded solve routine.

        Based on Algorithm 5.2 of [2]_ (but modified to avoid factorize)
        """
        device, dtype = Ta.device, Ta.dtype

        # convert to numpy
        Ta = Ta.cpu().numpy()
        Tb = Tb.cpu().numpy()
        tr_radius = float(tr_radius)

        # right hand side
        rhs = np.zeros_like(Ta)
        rhs[0] = - float(gamma_0)

        # estimate smallest eigenvalue
        eig0 = eigh_tridiagonal(
            Ta, Tb, eigvals_only=True, select='i',
            select_range=(0,0), lapack_driver='stebz')
        eig0 = eig0.item()
        lambd_lb = max(-eig0, 0) + 1e-3

        # get LAPACK solver for sym-PD banded matrices
        ptsv, = get_lapack_funcs(('ptsv',), (Ta, Tb, rhs))

        lambd = self.lambd_0
        for _ in range(self.max_ms_iters):
            lambd = max(lambd, lambd_lb)
            Ta_k = Ta + lambd
            _, _, p, info = ptsv(Ta_k, Tb, rhs)
            assert info == 0
            p_norm = np.linalg.norm(p)
            if p_norm < tr_radius:
                status = 0
                break
            elif abs(p_norm - tr_radius) / tr_radius <= self.k_easy:
                status = 1
                break
            _, _, v, info = ptsv(Ta_k, Tb, p)
            assert info == 0
            lambd += (p_norm**2 / v.dot(p)) * (p_norm - tr_radius) / tr_radius
        else:
            status = -1

        p = torch.tensor(p, device=device, dtype=dtype)

        return p, status, lambd


    def solve_krylov_mod1(self, Ta, Tb, gamma_0, tr_radius):
        """This version uses the unit bi-diagonal factorization schema for T

        Based on Algorithm 5.2 of [2]_
        """
        device, dtype = Ta.device, Ta.dtype

        # convert to numpy
        Ta = Ta.cpu().numpy()
        Tb = Tb.cpu().numpy()
        tr_radius = float(tr_radius)

        # right hand side
        rhs = np.zeros_like(Ta)
        rhs[0] = - float(gamma_0)

        # prepare functions for factorizing and solving sym-PD tridiagonal
        factor_td, solve_td = get_lapack_funcs(('pttrf', 'pttrs'), (Ta, Tb))

        def solve_unit_bidiag(ee, pp):
            raise NotImplementedError  # TODO

        # estimate smallest eigenvalue
        eig0 = eigh_tridiagonal(
            Ta, Tb, eigvals_only=True, select='i',
            select_range=(0,0), lapack_driver='stebz')
        eig0 = eig0.item()
        lambd_lb = max(-eig0, 0) + 1e-3

        lambd = self.lambd_0
        for _ in range(self.max_ms_iters):
            lambd = max(lambd, lambd_lb)
            d, e, info = factor_td(Ta+lambd, Tb)
            assert info == 0
            p, info = solve_td(d, e, rhs)
            assert info == 0
            p_norm = np.linalg.norm(p)
            if p_norm < tr_radius:
                # TODO: add extra checks
                status = 0
                break
            elif abs(p_norm - tr_radius) / tr_radius <= self.k_easy:
                status = 1
                break
            q = solve_unit_bidiag(e, p)  # TODO
            q_norm = np.linalg.norm(q)
            lambd += (p_norm / q_norm)**2 * (p_norm - tr_radius) / tr_radius
        else:
            status = -1

        p = torch.tensor(p, device=device, dtype=dtype)

        return p, status, lambd

    def solve_krylov(self, Ta, Tb, gamma_0, tr_radius):
        """Solve the trust-region sub-problem within a Krylov subspace

        Ta and Tb are the diagonal and off-diagonal parts of the (symmetric)
        tridiagonal matrix T. We use a variant of the Moré-Sorensen algorithm
        that exploits the tri-diagonal structure of T.
        """
        # eigen decomposition
        eig, V = eigh_tridiag(Ta, Tb)
        VT = V.T

        # right-hand side of linear sub-problem
        rhs = torch.zeros_like(Ta)
        rhs[0] = - gamma_0
        Vrhs = VT.mv(rhs)

        # lower-bound on lambda
        lambd_lb = eig[0].neg().clamp(min=0) + 1e-3

        # iterate
        lambd = torch.tensor(self.lambd_0, device=Ta.device, dtype=Ta.dtype)
        for _ in range(self.max_ms_iters):
            lambd.clamp_(min=lambd_lb)
            #x = solveh_tridiag(T + lambd * I, rhs, pos=True)
            eig_k = eig + lambd
            if self._debug:
                assert torch.all(eig_k >= 0), 'negative eigenvalue: %0.4e' % eig_k.min()
            p = V.mv(Vrhs / eig_k)
            p_norm = torch.linalg.norm(p)
            if p_norm < tr_radius:
                # TODO: add extra checks
                status = 0
                break
            elif torch.abs(p_norm - tr_radius) / tr_radius <= self.k_easy:
                status = 1
                break
            q = VT.mv(p) / eig_k.sqrt()
            q_norm = torch.linalg.norm(q)
            lambd.addcmul_((p_norm / q_norm)**2, (p_norm - tr_radius) / tr_radius)
        else:
            status = -1

        return p, status, lambd

    def solve(self, tr_radius):
        g = self.jac
        gamma_0 = self.jac_mag
        n, = g.shape
        m = n if self.max_lanczos is None else min(n, self.max_lanczos)
        dtype = g.dtype
        device = g.device
        hits_boundary = True

        # Lanczos Q matrix buffer
        Q = torch.zeros(m, n, dtype=dtype, device=device)
        Q[0] = g

        # Lanczos T matrix buffers
        # a and b are the diagonal and off-diagonal entries of T, respectively
        a = torch.zeros(m, dtype=dtype, device=device)
        b = torch.zeros(m, dtype=dtype, device=device)

        # first lanczos iteration
        if torch.abs(gamma_0 - 1) > self.eps:
            Q[0].div_(gamma_0)
        r = self.hessp(Q[0])
        torch.dot(Q[0], r, out=a[0])
        r.sub_(Q[0], alpha=a[0])

        # remaining iterations
        for i in range(1, m):
            torch.linalg.norm(r, out=b[i-1])  # gamma_{k-1} = norm(v_k)
            if b[i-1] < self.eps:
                # TODO: what do we do here? For now treating it as 'singular'
                raise RuntimeError('singular matrix')
                # m = i; break

            torch.div(r, b[i-1], out=Q[i])  # q
            r = self.hessp(Q[i])  # H @ q
            r.sub_(Q[i-1], alpha=b[i-1])  # v = H @ q - gamma_prev * q_prev
            torch.dot(Q[i], r, out=a[i])  # delta = <q, H @ q - gamma_prev * q_prev>
            r.sub_(Q[i], alpha=a[i])  # v = H @ q - gamma_prev * q_prev - delta * q
            if self.ortho:
                # re-orthogonalize
                r.addmv_(Q[:i+1].T, Q[:i+1].mv(r), alpha=-1)

            # GLTR sub-problem
            h, status, lambd = self.solve_krylov_mod2(a[:i+1], b[:i], gamma_0, tr_radius)

            if status >= 0:
                # project p back to R^n
                p = Q[:i+1].T.mv(h)
                # convergence check; see Algorithm 1 of [1]_
                g_hat = self.hessp(p) + lambd * p
                rel_error = torch.linalg.norm(g_hat + g)
                if self._debug:
                    print('iter %3d - status: %d - lambd: %0.4e - p_norm: %0.4e'
                          ' - error: %0.4e' %
                          (i+1, status, lambd, p.norm(), rel_error))
                if rel_error <= self.tol:
                    hits_boundary = status != 0
                    break
        else:
            # TODO: what should we do here?
            p = -g
            hits_boundary = True

        return p, hits_boundary

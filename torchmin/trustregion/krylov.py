"""
TODO: this module is not yet complete. It is not ready for use.
"""
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
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    """
    hess_prod = True

    # extra variable defs
    tol = 1e-5
    lambd_0 = 1e-3
    max_lanczos = None
    max_ms_iters = 500  # max iterations of the Moré-Sorensen loop

    def __init__(self, x, fun, k_easy=0.1, k_hard=0.2, ortho=True, debug=False):
        super().__init__(x, fun)
        self.eps = torch.finfo(x.dtype).eps
        self.k_easy = k_easy
        self.k_hard = k_hard
        self.ortho = ortho
        self.nlanczos = 0
        self.best_obj = float('inf')
        self._debug = debug

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
        if self._debug:
            print('lambda_lb: %0.4e' % lambd_lb, end=' - ')

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
                if self._debug:
                    print('nlanczos=%4d: solution found' % self.nlanczos)
                # TODO: add extra checks
                status = 0
                break
            elif torch.abs(p_norm - tr_radius) / tr_radius <= self.k_easy:
                if self._debug:
                    print('nlanczos=%4d: relative error reached' % self.nlanczos)
                status = 1
                break
            q = VT.mv(p) / eig_k.sqrt()
            q_norm = torch.linalg.norm(q)
            lambd.addcmul_((p_norm / q_norm)**2, (p_norm - tr_radius) / tr_radius)
        else:
            if self._debug:
                print('nlanczos=%d: krylov search did not converge' % self.nlanczos)
            status = -1

        return p, status, lambd

    def solve(self, tr_radius):
        g = self.jac
        gamma_0 = self.jac_mag
        n, = g.shape
        m = n if self.max_lanczos is None else min(n, self.max_lanczos)
        dtype = g.dtype
        device = g.device
        self.nlanczos = 0
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
        self.nlanczos += 1

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

            self.nlanczos += 1

            # GLTR sub-problem
            h, status, lambd = self.solve_krylov(a[:i+1], b[:i], gamma_0, tr_radius)

            if status >= 0:
                # project p back to R^n
                p = Q[:i+1].T.mv(h)
                if self._debug:
                    print('objective: %0.4e - p_norm: %0.4e' % (self(p), p.norm()))
                # convergence check; see Algorithm 1 of [1]_
                g_hat = self.hessp(p) + lambd * p
                if torch.linalg.norm(g_hat + g) <= self.tol:
                    hits_boundary = status != 0
                    break
        else:
            # TODO: what should we do here?
            p = -g
            hits_boundary = True

        return p, hits_boundary

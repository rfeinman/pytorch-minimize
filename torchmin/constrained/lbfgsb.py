import torch
from torch import Tensor
from scipy.optimize import OptimizeResult

from .._optimize import _status_message
from ..function import ScalarFunction


class L_BFGS_B:
    """Limited-memory BFGS Hessian approximation for bounded optimization.

    This class maintains the L-BFGS history and provides methods for
    computing search directions within bound constraints.
    """
    def __init__(self, x, history_size=10):
        self.y = []
        self.s = []
        self.rho = []
        self.theta = 1.0  # scaling factor
        self.history_size = history_size
        self.n_updates = 0

    def solve(self, grad, x, lb, ub, theta=None):
        """Compute search direction: -H * grad, respecting bounds.

        Parameters
        ----------
        grad : Tensor
            Current gradient
        x : Tensor
            Current point
        lb : Tensor
            Lower bounds
        ub : Tensor
            Upper bounds
        theta : float, optional
            Scaling factor. If None, uses stored value.

        Returns
        -------
        d : Tensor
            Search direction
        """
        if theta is not None:
            self.theta = theta

        mem_size = len(self.y)
        if mem_size == 0:
            # No history yet, use scaled steepest descent
            return grad.neg() * self.theta

        # Two-loop recursion
        alpha = torch.zeros(mem_size, dtype=grad.dtype, device=grad.device)
        q = grad.clone()

        # First loop: backward pass
        for i in reversed(range(mem_size)):
            alpha[i] = self.rho[i] * self.s[i].dot(q)
            q.add_(self.y[i], alpha=-alpha[i])

        # Apply initial Hessian approximation
        r = q * self.theta

        # Second loop: forward pass
        for i in range(mem_size):
            beta = self.rho[i] * self.y[i].dot(r)
            r.add_(self.s[i], alpha=alpha[i] - beta)

        return -r

    def update(self, s, y):
        """Update the L-BFGS history with new correction pair.

        Parameters
        ----------
        s : Tensor
            Step vector (x_new - x)
        y : Tensor
            Gradient difference (g_new - g)
        """
        # Check curvature condition
        sy = s.dot(y)
        if sy <= 1e-10:
            # Skip update if curvature is too small
            return False

        yy = y.dot(y)

        # Update scaling factor (theta = s'y / y'y)
        if yy > 1e-10:
            self.theta = sy / yy

        # Update history
        if len(self.y) >= self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)

        self.y.append(y.clone())
        self.s.append(s.clone())
        self.rho.append(1.0 / sy)
        self.n_updates += 1

        return True


def _project_bounds(x, lb, ub):
    """Project x onto the box [lb, ub]."""
    return torch.clamp(x, lb, ub)


def _gradient_projection(x, g, lb, ub):
    """Compute the projected gradient.

    Returns the projected gradient and identifies the active set.
    """
    # Project gradient: if at bound and gradient points out, set to zero
    g_proj = g.clone()

    # At lower bound with positive gradient
    at_lb = (x <= lb + 1e-10) & (g > 0)
    g_proj[at_lb] = 0

    # At upper bound with negative gradient
    at_ub = (x >= ub - 1e-10) & (g < 0)
    g_proj[at_ub] = 0

    return g_proj


@torch.no_grad()
def _minimize_lbfgsb(
        fun, x0, bounds=None, lr=1.0, history_size=10,
        max_iter=None, gtol=1e-5, ftol=1e-9,
        normp=float('inf'), callback=None, disp=0, return_all=False):
    """Minimize a scalar function with L-BFGS-B.

    L-BFGS-B [1]_ is a limited-memory quasi-Newton method for bound-constrained
    optimization. It extends L-BFGS to handle box constraints.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    bounds : tuple of Tensor, optional
        Bounds for variables as (lb, ub) where lb and ub are Tensors
        of the same shape as x0. Use float('-inf') and float('inf')
        for unbounded variables. If None, equivalent to unbounded.
    lr : float
        Step size for parameter updates (used as initial step in line search).
    history_size : int
        History size for L-BFGS Hessian estimates.
    max_iter : int, optional
        Maximum number of iterations. Defaults to 200 * x0.numel().
    gtol : float
        Termination tolerance on projected gradient norm.
    ftol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        Norm type for termination conditions. Can be any value
        supported by torch.norm.
    callback : callable, optional
        Function to call after each iteration: callback(x).
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each iteration.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    References
    ----------
    .. [1] Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory
       algorithm for bound constrained optimization. SIAM Journal on
       Scientific Computing, 16(5), 1190-1208.
    """
    lr = float(lr)
    disp = int(disp)
    if max_iter is None:
        max_iter = x0.numel() * 200

    # Set up bounds
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    n = x.numel()

    if bounds is None:
        lb = torch.full_like(x, float('-inf'))
        ub = torch.full_like(x, float('inf'))
    else:
        lb, ub = bounds
        lb = lb.detach().view(-1).clone(memory_format=torch.contiguous_format)
        ub = ub.detach().view(-1).clone(memory_format=torch.contiguous_format)

        if lb.shape != x.shape or ub.shape != x.shape:
            raise ValueError('Bounds must have the same shape as x0')

    # Project initial point onto feasible region
    x = _project_bounds(x, lb, ub)

    # Construct scalar objective function
    sf = ScalarFunction(fun, x0.shape)
    closure = sf.closure

    # Compute initial function and gradient
    f, g, _, _ = closure(x)

    if disp > 1:
        print('initial fval: %0.4f' % f)
        print('initial gnorm: %0.4e' % g.norm(p=normp))

    if return_all:
        allvecs = [x.clone()]

    # Initialize L-BFGS approximation
    hess = L_BFGS_B(x, history_size)

    # Main iteration loop
    for n_iter in range(1, max_iter + 1):

        # ========================================
        #   Check projected gradient convergence
        # ========================================

        g_proj = _gradient_projection(x, g, lb, ub)
        g_proj_norm = g_proj.norm(p=normp)

        if disp > 1:
            print('iter %3d - fval: %0.4f, gnorm: %0.4e' %
                  (n_iter, f, g_proj_norm))

        if g_proj_norm <= gtol:
            warnflag = 0
            msg = _status_message['success']
            break

        # ========================================
        #   Compute search direction
        # ========================================

        # Use projected gradient for search direction computation
        # This ensures we only move in directions away from active constraints
        d = hess.solve(g_proj, x, lb, ub)

        # Ensure direction is a descent direction w.r.t. original gradient
        gtd = g.dot(d)
        if gtd > -1e-10:
            # Not a descent direction, use projected steepest descent
            d = -g_proj
            gtd = g.dot(d)

        # Find maximum step length that keeps us feasible
        alpha_max = 1.0
        for i in range(x.numel()):
            if d[i] > 1e-10:
                # Moving toward upper bound
                if ub[i] < float('inf'):
                    alpha_max = min(alpha_max, (ub[i] - x[i]) / d[i])
            elif d[i] < -1e-10:
                # Moving toward lower bound
                if lb[i] > float('-inf'):
                    alpha_max = min(alpha_max, (lb[i] - x[i]) / d[i])

        # Take a step with line search on the feasible segment
        # Simple backtracking: try alpha_max, 0.5*alpha_max, etc.
        alpha = alpha_max
        for _ in range(10):
            x_new = x + alpha * d
            x_new = _project_bounds(x_new, lb, ub)
            f_new, g_new, _, _ = closure(x_new)

            # Armijo condition (sufficient decrease)
            if f_new <= f + 1e-4 * alpha * gtd:
                break
            alpha *= 0.5
        else:
            # Line search failed, take a small step
            x_new = x + 0.01 * alpha_max * d
            x_new = _project_bounds(x_new, lb, ub)
            f_new, g_new, _, _ = closure(x_new)

        if return_all:
            allvecs.append(x_new.clone())

        if callback is not None:
            if callback(x_new.view_as(x0)):
                warnflag = 5
                msg = _status_message['callback_stop']
                break

        # ========================================
        #   Update Hessian approximation
        # ========================================

        s = x_new - x
        y = g_new - g

        # Update L-BFGS (skip if curvature condition fails)
        hess.update(s, y)

        # ========================================
        #   Check convergence by small progress
        # ========================================

        # Convergence by insufficient progress (be more lenient than gtol)
        if (s.norm(p=normp) <= ftol) and ((f_new - f).abs() <= ftol):
            # Double check with projected gradient
            g_proj_new = _gradient_projection(x_new, g_new, lb, ub)
            if g_proj_new.norm(p=normp) <= gtol:
                warnflag = 0
                msg = _status_message['success']
                break

        # Check for precision loss
        if not f_new.isfinite():
            warnflag = 2
            msg = _status_message['pr_loss']
            break

        # Update state
        f = f_new
        x = x_new
        g = g_new

    else:
        # Maximum iterations reached
        warnflag = 1
        msg = _status_message['maxiter']

    if disp:
        print(msg)
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)

    result = OptimizeResult(
        fun=f,
        x=x.view_as(x0),
        grad=g.view_as(x0),
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        nit=n_iter,
        nfev=sf.nfev
    )

    if return_all:
        result['allvecs'] = [v.view_as(x0) for v in allvecs]

    return result

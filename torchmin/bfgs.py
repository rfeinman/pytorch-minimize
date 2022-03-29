from abc import ABC, abstractmethod
import torch
from torch import Tensor
from scipy.optimize import OptimizeResult

from .function import ScalarFunction
from .line_search import strong_wolfe

try:
    from scipy.optimize.optimize import _status_message
except ImportError:
    from scipy.optimize._optimize import _status_message

class HessianUpdateStrategy(ABC):
    def __init__(self):
        self.n_updates = 0

    @abstractmethod
    def solve(self, grad):
        pass

    @abstractmethod
    def _update(self, s, y, rho_inv):
        pass

    def update(self, s, y):
        rho_inv = y.dot(s)
        if rho_inv <= 1e-10:
            # curvature is negative; do not update
            return
        self._update(s, y, rho_inv)
        self.n_updates += 1


class L_BFGS(HessianUpdateStrategy):
    def __init__(self, x, history_size=100):
        super().__init__()
        self.y = []
        self.s = []
        self.rho = []
        self.H_diag = 1.
        self.alpha = x.new_empty(history_size)
        self.history_size = history_size

    def solve(self, grad):
        mem_size = len(self.y)
        d = grad.neg()
        for i in reversed(range(mem_size)):
            self.alpha[i] = self.s[i].dot(d) * self.rho[i]
            d.add_(self.y[i], alpha=-self.alpha[i])
        d.mul_(self.H_diag)
        for i in range(mem_size):
            beta_i = self.y[i].dot(d) * self.rho[i]
            d.add_(self.s[i], alpha=self.alpha[i] - beta_i)

        return d

    def _update(self, s, y, rho_inv):
        if len(self.y) == self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)
        self.y.append(y)
        self.s.append(s)
        self.rho.append(rho_inv.reciprocal())
        self.H_diag = rho_inv / y.dot(y)


class BFGS(HessianUpdateStrategy):
    def __init__(self, x, inverse=True):
        super().__init__()
        self.inverse = inverse
        if inverse:
            self.I = torch.eye(x.numel(), device=x.device, dtype=x.dtype)
            self.H = self.I.clone()
        else:
            self.B = torch.eye(x.numel(), device=x.device, dtype=x.dtype)

    def solve(self, grad):
        if self.inverse:
            return torch.matmul(self.H, grad.neg())
        else:
            return torch.cholesky_solve(grad.neg().unsqueeze(1),
                                        torch.linalg.cholesky(self.B)).squeeze(1)

    def _update(self, s, y, rho_inv):
        rho = rho_inv.reciprocal()
        if self.inverse:
            if self.n_updates == 0:
                self.H.mul_(rho_inv / y.dot(y))
            R = torch.addr(self.I, s, y, alpha=-rho)
            torch.addr(
                torch.linalg.multi_dot((R, self.H, R.t())),
                s, s, alpha=rho, out=self.H)
        else:
            if self.n_updates == 0:
                self.B.mul_(rho * y.dot(y))
            Bs = torch.mv(self.B, s)
            self.B.addr_(y, y, alpha=rho)
            self.B.addr_(Bs, Bs, alpha=-1./s.dot(Bs))


@torch.no_grad()
def _minimize_bfgs_core(
        fun, x0, lr=1., low_mem=False, history_size=100, inv_hess=True,
        max_iter=None, line_search='strong-wolfe', gtol=1e-5, xtol=1e-9,
        normp=float('inf'), callback=None, disp=0, return_all=False):
    """Minimize a multivariate function with BFGS or L-BFGS.

    We choose from BFGS/L-BFGS with the `low_mem` argument.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    low_mem : bool
        Whether to use L-BFGS, the "low memory" variant of the BFGS algorithm.
    history_size : int
        History size for L-BFGS hessian estimates. Ignored if `low_mem=False`.
    inv_hess : bool
        Whether to parameterize the inverse hessian vs. the hessian with BFGS.
        Ignored if `low_mem=True` (L-BFGS always parameterizes the inverse).
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * x0.numel()
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    xtol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by `torch.norm` p argument.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)
    if max_iter is None:
        max_iter = x0.numel() * 200
    if low_mem and not inv_hess:
        raise ValueError('inv_hess=False is not available for L-BFGS.')

    # construct scalar objective function
    sf = ScalarFunction(fun, x0.shape)
    closure = sf.closure
    if line_search == 'strong-wolfe':
        dir_evaluate = sf.dir_evaluate

    # compute initial f(x) and f'(x)
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    f, g, _, _ = closure(x)
    if disp > 1:
        print('initial fval: %0.4f' % f)
    if return_all:
        allvecs = [x]

    # initial settings
    if low_mem:
        hess = L_BFGS(x, history_size)
    else:
        hess = BFGS(x, inv_hess)
    d = g.neg()
    t = min(1., g.norm(p=1).reciprocal()) * lr
    n_iter = 0

    # BFGS iterations
    for n_iter in range(1, max_iter+1):

        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            d = hess.solve(g)

        # directional derivative
        gtd = g.dot(d)

        # check if directional derivative is below tolerance
        if gtd > -xtol:
            warnflag = 4
            msg = 'A non-descent direction was encountered.'
            break

        # ======================
        #   update parameter
        # ======================

        if line_search == 'none':
            # no line search, move with fixed-step
            x_new = x + d.mul(t)
            f_new, g_new, _, _ = closure(x_new)
        elif line_search == 'strong-wolfe':
            #  Determine step size via strong-wolfe line search
            f_new, g_new, t, ls_evals = \
                strong_wolfe(dir_evaluate, x, t, d, f, g, gtd)
            x_new = x + d.mul(t)
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (n_iter, f_new))
        if return_all:
            allvecs.append(x_new)
        if callback is not None:
            callback(x_new)

        # ================================
        #   update hessian approximation
        # ================================

        s = x_new.sub(x)
        y = g_new.sub(g)

        hess.update(s, y)

        # =========================================
        #   check conditions and update buffers
        # =========================================

        # convergence by insufficient progress
        if (s.norm(p=normp) <= xtol) | ((f_new - f).abs() <= xtol):
            warnflag = 0
            msg = _status_message['success']
            break

        # update state
        f[...] = f_new
        x.copy_(x_new)
        g.copy_(g_new)
        t = lr

        # convergence by 1st-order optimality
        if g.norm(p=normp) <= gtol:
            warnflag = 0
            msg = _status_message['success']
            break

        # precision loss; exit
        if ~f.isfinite():
            warnflag = 2
            msg = _status_message['pr_loss']
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1
        msg = _status_message['maxiter']

    if disp:
        print(msg)
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
    result = OptimizeResult(fun=f, x=x.view_as(x0), grad=g.view_as(x0),
                            status=warnflag, success=(warnflag==0),
                            message=msg, nit=n_iter, nfev=sf.nfev)
    if not low_mem:
        if inv_hess:
            result['hess_inv'] = hess.H.view(2 * x0.shape)
        else:
            result['hess'] = hess.B.view(2 * x0.shape)
    if return_all:
        result['allvecs'] = allvecs

    return result


def _minimize_bfgs(
        fun, x0, lr=1., inv_hess=True, max_iter=None,
        line_search='strong-wolfe', gtol=1e-5, xtol=1e-9,
        normp=float('inf'), callback=None, disp=0, return_all=False):
    """Minimize a multivariate function with BFGS

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    inv_hess : bool
        Whether to parameterize the inverse hessian vs. the hessian with BFGS.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    xtol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    return _minimize_bfgs_core(
        fun, x0, lr, low_mem=False, inv_hess=inv_hess, max_iter=max_iter,
        line_search=line_search, gtol=gtol, xtol=xtol,
        normp=normp, callback=callback, disp=disp, return_all=return_all)


def _minimize_lbfgs(
        fun, x0, lr=1., history_size=100, max_iter=None,
        line_search='strong-wolfe', gtol=1e-5, xtol=1e-9,
        normp=float('inf'), callback=None, disp=0, return_all=False):
    """Minimize a multivariate function with L-BFGS

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    history_size : int
        History size for L-BFGS hessian estimates.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    xtol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    return _minimize_bfgs_core(
        fun, x0, lr, low_mem=True, history_size=history_size,
        max_iter=max_iter, line_search=line_search, gtol=gtol, xtol=xtol,
        normp=normp, callback=callback, disp=disp, return_all=return_all)
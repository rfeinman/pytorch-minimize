from abc import ABC, abstractmethod
import warnings
import math
import torch
from torch import Tensor
from scipy.optimize import OptimizeResult

from .function import ScalarFunction
from .line_search import strong_wolfe, scipy_wolfe

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

    def update(self, s, y, alpha_k=None, gfk=None):
        rho_inv = y.dot(s)
        # Only skip update if curvature condition is violated (rho_inv <= 0)
        # Use a very small threshold to handle numerical precision issues
        if rho_inv <= 1e-20:
            # curvature is negative or too small; do not update
            return
        self._update(s, y, rho_inv, alpha_k=alpha_k, gfk=gfk)
        self.n_updates += 1


class L_SSBroyden(HessianUpdateStrategy):
    """Limited-memory SSBroyden (L-SSBroyden) update strategy."""
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

    def _update(self, s, y, rho_inv, alpha_k=None, gfk=None):
        if len(self.y) == self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)
        self.y.append(y)
        self.s.append(s)
        self.rho.append(rho_inv.reciprocal())
        self.H_diag = rho_inv / y.dot(y)


class SSBroyden(HessianUpdateStrategy):
    """SSBroyden2 update strategy for inverse Hessian approximation."""
    def __init__(self, x, initial_scale=False, hess_inv0=None):
        super().__init__()
        self.initial_scale = initial_scale
        self.I = torch.eye(x.numel(), device=x.device, dtype=x.dtype)
        if hess_inv0 is not None:
            self.H = torch.as_tensor(hess_inv0, device=x.device, dtype=x.dtype)
        else:
            self.H = self.I.clone()

    def solve(self, grad):
        return torch.matmul(self.H, grad.neg())

    def _update(self, s, y, rho_inv, alpha_k=None, gfk=None):
        """SSBroyden2 update formula based on SciPy implementation."""
        N = self.H.shape[0]
        rhok = rho_inv.reciprocal()
        Hkyk = torch.matmul(self.H, y)
        ykHkyk = y.dot(Hkyk)
        hk = ykHkyk * rhok
        
        # Compute bk: bk = -alpha_k*rhok*s.dot(gfk)
        if alpha_k is not None and gfk is not None:
            bk = -alpha_k * rhok * s.dot(gfk)
        else:
            # Fallback if alpha_k/gfk not available
            bk = -s.dot(y) * rhok
        
        # Compute ak = hk*bk - 1
        ak = hk * bk - 1
        
        # Numerical stability checks
        ak_val = ak.item() if hasattr(ak, 'item') else float(ak)
        one_plus_ak = 1 + ak_val
        
        if one_plus_ak <= 1e-12:
            rhokm = 1.0
        else:
            ak_ratio = max(0.0, min(1.0, abs(ak_val) / one_plus_ak))
            hk_val = hk.item() if hasattr(hk, 'item') else float(hk)
            rhokm = max(1e-12, min(1.0, hk_val * (1 - math.sqrt(ak_ratio))))
        
        # Compute thetak
        thetakm = (rhokm - 1) / ak_val if abs(ak_val) > 1e-12 else 0.0
        thetakp = 1 / rhokm
        bk_val = bk.item() if hasattr(bk, 'item') else float(bk)
        
        if abs(bk_val) < 1e-12:
            thetak = thetakm
        else:
            thetak = max(thetakm, min(thetakp, (1 - bk_val) / bk_val))
        
        # Compute tauk
        if self.initial_scale and self.n_updates == 0 and torch.allclose(self.H, self.I):
            tauk = hk / (1 + ak * thetak)
        else:
            rhokk = min(1.0, 1 / bk)
            sigmak = 1 + thetak * ak
            sigmaknm1 = torch.pow(torch.abs(sigmak), 1.0 / (1 - N))
            
            if thetak <= 0:
                tauk = min(rhokk * sigmaknm1, sigmak)
            else:
                tauk = rhokk * min(sigmaknm1, 1 / thetak)
        
        # SSBroyden2 update: Hk = (Hk - Hkyk*Hkyk^T/ykHkyk + phik*ykHkyk*vk*vk^T)/tauk + sk*sk^T*rhok
        vk = s * rhok - Hkyk / ykHkyk
        phik = (1 - thetak) / (1 + ak * thetak)
        
        Hkyk_col = Hkyk.unsqueeze(1)
        Hkyk_row = Hkyk.unsqueeze(0)
        vk_col = vk.unsqueeze(1)
        vk_row = vk.unsqueeze(0)
        s_col = s.unsqueeze(1)
        s_row = s.unsqueeze(0)
        
        self.H.sub_((Hkyk_col @ Hkyk_row) / ykHkyk)
        self.H.add_(phik * ykHkyk * (vk_col @ vk_row))
        self.H.div_(tauk)
        self.H.add_(rhok * (s_col @ s_row))


@torch.no_grad()
def _minimize_ssbroyden_core(
        fun, x0, lr=1.e-3, low_mem=False, history_size=100,
        max_iter=None, line_search='scipy-wolfe', gtol=1e-5, xtol=1e-9,
        ftol=1e-8, xrtol=0, normp=float('inf'), callback=None, disp=0, return_all=False,
        initial_scale=True, hess_inv0=None):
    """Minimize a multivariate function with SSBroyden or L-SSBroyden.

    We choose from SSBroyden/L-SSBroyden with the `low_mem` argument.

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
        Whether to use L-SSBroyden, the "low memory" variant of the SSBroyden algorithm.
    history_size : int
        History size for L-SSBroyden hessian estimates. Ignored if `low_mem=False`.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * x0.numel()
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe', 'scipy-wolfe'}.
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

    # construct scalar objective function
    sf = ScalarFunction(fun, x0.shape)
    closure = sf.closure
    if line_search in ['strong-wolfe', 'scipy-wolfe']:
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
        hess = L_SSBroyden(x, history_size)
    else:
        hess = SSBroyden(x, initial_scale=initial_scale, hess_inv0=hess_inv0)
    d = g.neg()
    # SciPy-style step size initialization with better scaling
    if g.norm() > 0:
        t = min(1., 1.0 / g.norm()) * lr
    else:
        t = lr
    n_iter = 0

    # SSBroyden iterations
    for n_iter in range(1, max_iter+1):
        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            d = hess.solve(g)
        else:
            d = g.neg()  # First iteration uses steepest descent

        # directional derivative
        gtd = g.dot(d)
        
        # # Debug: print state BEFORE line search (at start of iteration)
        # print(f"DEBUG: Iteration {n_iter}/{max_iter} - BEFORE step: f={f.item():.2e}, ||g||={g.norm().item():.2e}, ||d||={d.norm().item():.2e}, gtd={gtd.item():.2e}")

        # Note: SciPy doesn't check for non-descent directions here.
        # The line search will handle finding an appropriate step size.
        # If the line search fails, it will raise an exception.

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
        elif line_search == 'scipy-wolfe':
            #  Determine step size via SciPy's _line_search_wolfe12 (wolfe1 then wolfe2)
            f_new, g_new, t, ls_evals = \
                scipy_wolfe(dir_evaluate, x, t, d, f, g, gtd)
            x_new = x + d.mul(t)
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        t_val = t.item() if hasattr(t, 'item') else t

        if return_all:
            allvecs.append(x_new)
        if callback is not None:
            callback(x_new)

        # ================================
        #   update hessian approximation
        # ================================

        s = x_new.sub(x)
        y = g_new.sub(g)
        
        # Pass alpha_k (step size) and gfk (old gradient) for SSBroyden2
        hess.update(s, y, alpha_k=t.item() if hasattr(t, 'item') else t, gfk=g)

        # =========================================
        #   check conditions and update buffers
        # =========================================

        # Debug: print step results AFTER line search (before state update)
        # f_change shows change from OLD f to NEW f (should be negative for descent)
        s_norm = s.norm(p=normp).item()
        f_new_val = f_new.item() if hasattr(f_new, 'item') else f_new
        f_change = f_new_val - f.item()  # Show actual change (negative = descent)
        g_norm = g_new.norm(p=normp).item()
        

        # SciPy-style relative tolerance check (moved up for better performance)
        if xrtol > 0 and t * d.norm() <= xrtol * (xrtol + x.norm()):
            warnflag = 0
            msg = _status_message['success']
            break

        # convergence by insufficient progress
        f_diff_abs = abs(f_change)  # f_change already computed above as f_new_val - f.item()
        f_val = abs(f.item()) if hasattr(f, 'item') else abs(float(f))
        
        # Check parameter change tolerance
        param_change = (s.norm(p=normp).item() if hasattr(s.norm(p=normp), 'item') else float(s.norm(p=normp))) <= xtol
        
        # Check absolute function change tolerance (use xtol for backward compatibility)
        func_change_abs = f_diff_abs <= xtol
        
        # Check relative function tolerance (SciPy-style: dF < ftol * F)
        # Only check if current function value is not too small to avoid division issues
        func_change_rel = False
        if f_val > 1e-12:  # Avoid division by very small numbers
            func_change_rel = f_diff_abs <= ftol * f_val
        
        if param_change or func_change_abs or func_change_rel:
            warnflag = 0
            msg = _status_message['success']
            break

        # update state
        if isinstance(f_new_val, (int, float)) and not hasattr(f_new, 'item'):
            f[...] = torch.tensor(f_new_val, dtype=f.dtype, device=f.device)
        else:
            f[...] = f_new
        x.copy_(x_new)
        g.copy_(g_new)
        # SciPy-style step size update with better scaling
        if g.norm() > 0:
            t = min(1., 1.0 / g.norm()) * lr
        else:
            t = lr

        # convergence by 1st-order optimality (moved after state update for efficiency)
        gnorm = g.norm(p=normp)
        if gnorm <= gtol:
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
        result['hess_inv'] = hess.H.view(2 * x0.shape)
    if return_all:
        result['allvecs'] = allvecs

    return result


def _minimize_ssbroyden(
        fun, x0, lr=1.e-3, max_iter=None,
        line_search='scipy-wolfe', gtol=1e-5, xtol=1e-9, ftol=1e-8, xrtol=0,
        normp=float('inf'), callback=None, disp=0, return_all=False,
        initial_scale=True, hess_inv0=None):
    """Minimize a multivariate function with SSBroyden

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong-wolfe', 'scipy-wolfe'}.
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
    initial_scale : bool
        Whether to use initial scaling for the Hessian approximation.
    hess_inv0 : Tensor, optional
        Initial inverse Hessian approximation.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    return _minimize_ssbroyden_core(
        fun, x0, lr, low_mem=False, max_iter=max_iter,
        line_search=line_search, gtol=gtol, xtol=xtol, ftol=ftol, xrtol=xrtol,
        normp=normp, callback=callback, disp=disp, return_all=return_all,
        initial_scale=initial_scale, hess_inv0=hess_inv0)


def _minimize_lssbroyden(
        fun, x0, lr=1., history_size=100, max_iter=None,
        line_search='strong-wolfe', gtol=1e-5, xtol=1e-9,
        normp=float('inf'), callback=None, disp=0, return_all=False):
    """Minimize a multivariate function with L-SSBroyden

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
        History size for L-SSBroyden hessian estimates.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong-wolfe', 'scipy-wolfe'}.
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
    return _minimize_ssbroyden_core(
        fun, x0, lr, low_mem=True, history_size=history_size,
        max_iter=max_iter, line_search=line_search, gtol=gtol, xtol=xtol,
        normp=normp, callback=callback, disp=disp, return_all=return_all)
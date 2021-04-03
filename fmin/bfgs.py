from abc import ABC, abstractmethod
import torch
from torch import Tensor
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message

from .line_search import strong_wolfe


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
                                        torch.cholesky(self.B)).squeeze(1)

    def _update(self, s, y, rho_inv):
        rho = rho_inv.reciprocal()
        if self.inverse:
            torch.addr(
                torch.chain_matmul(
                    torch.addr(self.I, s, y, alpha=-rho),
                    self.H,
                    torch.addr(self.I, y, s, alpha=-rho)
                ),
                s, s, alpha=rho, out=self.H
            )
        else:
            Bs = torch.mv(self.B, s)
            torch.addr(
                torch.addr(self.B, y, y, alpha=rho),
                Bs, Bs,
                alpha=s.dot(Bs).reciprocal().neg(),
                out=self.B
            )


@torch.no_grad()
def fmin_bfgs(
        f, x0, lr=1., low_mem=False, history_size=100, inv_hess=True,
        max_iter=None, line_search='strong-wolfe', gtol=1e-5, xtol=1e-9,
        normp=float('inf'), callback=None, disp=0, return_all=False):
    """Minimize a multivariate function with BFGS or L-BFGS

    Parameters
    ----------
    f : callable
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
        Termination tolerance on 1st-order optimality (gradient magnitude)
    xtol : float
        Termination tolerance on function/parameter changes
    normp : int | float | str
        The norm type to use for termination conditions. Can be any value
        supported by `torch.norm` p argument.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. callback(x_k)
    disp : int | bool
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

    def f_with_grad(x):
        x = x.view_as(x0).detach().requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        grad, = torch.autograd.grad(fval, x)
        return fval.detach(), grad.view(-1)

    def dir_evaluate(x, t, d):
        x = x.add(d, alpha=t)
        return f_with_grad(x)

    # compute initial f(x) and f'(x)
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    fval, grad = f_with_grad(x)
    nfev = 1
    if disp > 1:
        print('initial fval: %0.4f' % fval)
    if return_all:
        allvecs = [x]

    # initial settings
    if low_mem:
        hess = L_BFGS(x, history_size)
    else:
        hess = BFGS(x, inv_hess)
    d = grad.neg()
    t = min(1., grad.norm(p=1).reciprocal()) * lr

    # termination func
    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % n_iter)
            print("         Function evaluations: %d" % nfev)
        result = OptimizeResult(fun=fval, jac=grad, nfev=nfev,
                                status=warnflag, success=(warnflag==0),
                                message=msg, x=x.view_as(x0), nit=n_iter)
        if return_all:
            result['allvecs'] = allvecs
        return result

    # initial convergence check
    if grad.norm(p=normp) <= gtol:
        return terminate(0, _status_message['success'])

    # BFGS iterations
    for n_iter in range(1, max_iter+1):

        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            d = hess.solve(grad)

        # directional derivative
        gtd = grad.dot(d)

        # check if directional derivative is below tolerance
        # TODO: is this a success?
        if gtd > -xtol:
            return terminate(0, _status_message['success'])


        # ======================
        #   update parameter
        # ======================

        if line_search == 'none':
            # no line search, move with fixed-step
            x_new = x.add(d, alpha=t)
            fval_new, grad_new = f_with_grad(x_new)
            nfev += 1
        elif line_search == 'strong-wolfe':
            #  Determine step size via strong-wolfe line search
            fval_new, grad_new, t, ls_evals = \
                strong_wolfe(dir_evaluate, x, t, d, fval, grad, gtd)
            x_new = x.add(d, alpha=t)
            nfev += ls_evals
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (n_iter, fval_new))
        if return_all:
            allvecs.append(x_new)
        if callback is not None:
            callback(x_new)


        # ================================
        #   update hessian approximation
        # ================================

        s = x_new.sub(x)
        y = grad_new.sub(grad)

        hess.update(s, y)


        # =========================================
        #   check conditions and update buffers
        # =========================================

        # convergence by 1st-order optimality
        if grad.norm(p=normp) <= gtol:
            return terminate(0, _status_message['success'])

        # convergence by insufficient progress
        if s.norm(p=normp) <= xtol or (fval_new-fval).abs() <= xtol:
            return terminate(0, _status_message['success'])

        # precision loss; exit
        if not fval.isfinite():
            return terminate(2, _status_message['pr_loss'])

        # update state
        fval = fval_new
        x.copy_(x_new)
        grad.copy_(grad_new)
        t = lr

    # if we get to the end, the maximum num. iterations was reached
    return terminate(1, "Warning: " + _status_message['maxiter'])
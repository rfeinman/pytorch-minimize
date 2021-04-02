from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
from torch import Tensor
import torch
import torch.autograd as autograd

from .line_search import strong_wolfe
from .conjgrad import conjgrad

_status_message['cg_warn'] = "Warning: CG iterations didn't converge. The " \
                             "Hessian is not positive definite."


@torch.no_grad()
def fmin_newtoncg(
        f, x0, lr=1., max_iter=None, cg_max_iter=None,
        twice_diffable=True, line_search='strong_wolfe', xtol=1e-5,
        callback=None, disp=0, return_all=False):
    """
    Minimize a scalar function of one or more variables using the
    Newton Conjugate Gradient method.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    lr : float
        Initial step size (learning rate) for each line search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * num_params
    cg_max_iter : int, optional
        Maximum number of iterations per CG sub-problem. Recommended to leave
        this at the default of 20 * num_params
    max_ls : int
        Maximum number of steps per line search. If max_ls=0, then no line
        search is performed. Otherwise, it's recommended to leave this at
        the default value of 25
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. callback(x_k)
    disp : int
        Verbosity level. Set to > 0 to print convergence messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    twice_diffable : bool
        Whether to assume the function is twice continuously differentiable.
        If True, hessian-vector product computation will be much faster.
    """
    lr = float(lr)
    disp = int(disp)
    xtol = x0.numel() * xtol
    if max_iter is None:
        max_iter = x0.numel() * 200
    if cg_max_iter is None:
        cg_max_iter = x0.numel() * 20

    def f_with_grad(x):
        x = x.view_as(x0).detach().requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        grad, = autograd.grad(fval, x)
        return fval.detach(), grad.view(-1)

    def dir_evaluate(x, t, d):
        """directional evaluate. Used only for strong-wolfe line search"""
        return f_with_grad(x.add(d, alpha=t))

    # generalized dot product for CG that supports batch inputs
    # TODO: let the user specify dot fn?
    dot = lambda u,v: u.mul(v).sum(-1, keepdim=True)

    # initial settings
    x = x0.detach()
    if return_all:
        allvecs = [x]
    fval = x.new_tensor(-1.)
    grad = x.new_full(x.shape, -1.)
    nfev = 0  # number of function evals
    ncg = 0   # number of cg iterations

    if disp > 1:
        fval = f(x)
        print('initial fval: %0.4f' % fval)

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % n_iter)
            print("         Function evaluations: %d" % nfev)
            print("         CG iterations: %d" % ncg)
        result = OptimizeResult(fun=fval, jac=grad, nfev=nfev, ncg=ncg,
                                status=warnflag, success=(warnflag==0),
                                message=msg, x=x, nit=n_iter)
        if return_all:
            result['allvecs'] = allvecs
        return result


    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ============================================================
        #  Compute a search direction pk by applying the CG method to
        #       H_f(xk) p = - J_f(xk) starting from 0.
        # ============================================================

        # Compute f(xk) and f'(xk)
        x.requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
            # Note: create graph of gradient to enable hessian-vector product
            grad, = autograd.grad(fval, x, create_graph=True)
        nfev += 1

        # Initialize hessian-vector product function.
        # Note: PyTorch `hvp` is significantly slower than `vhp` due to
        # backward mode AD constraints. If the function is twice continuously
        # differentiable, then hvp = vhp^T, and we can use the faster vhp.
        if twice_diffable:
            hvp = lambda v: autograd.grad(grad, x, v, retain_graph=True)[0]
        else:
            g_grad = torch.zeros_like(grad, requires_grad=True)
            with torch.enable_grad():
                g_x = autograd.grad(grad, x, g_grad, create_graph=True)[0]
            hvp = lambda v: autograd.grad(g_x, g_grad, v, retain_graph=True)[0]

        # Compute search direction with conjugate gradient (GG)
        d, cg_iters, cg_status = conjgrad(
            b=grad.detach().neg(), Adot=hvp, dot=dot,
            max_iter=cg_max_iter, rtol=1., return_info=True
        )
        ncg += cg_iters
        if cg_status == 4:
            return terminate(3, _status_message['cg_warn'])

        # Free the autograd graph
        x = x.detach()
        fval = fval.detach()
        grad = grad.detach()


        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == 'none':
            update = d.mul(lr)
            x = x + update
            fval = f(x)
        elif line_search == 'strong_wolfe':
            # strong-wolfe line search
            gtd = grad.mul(d).sum()
            fval, grad, t, ls_nevals = \
                strong_wolfe(dir_evaluate, x.view(-1), lr, d.view(-1), fval,
                             grad.view(-1), gtd)
            grad = grad.view_as(x)
            nfev += ls_nevals
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError('invalid line_search option {}.'.format(line_search))

        if disp > 1:
            print('iter %3d - fval: %0.4f' % (n_iter, fval))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)


        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=1) <= xtol:
            return terminate(0, _status_message['success'])

        if not fval.isfinite():
            return terminate(3, _status_message['nan'])

    # if we get to the end, the maximum num. iterations was reached
    return terminate(1, "Warning: " + _status_message['maxiter'])
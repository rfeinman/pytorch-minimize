# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 16:28:41 2022

@author: X226840
"""

from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
import torch

from .function import ScalarFunction

@torch.no_grad()
def _minimize_lm_exact(
        fun, x0, max_iter=None, xtol=1e-5, normp=1, callback=None, disp=0,
        return_all=False, attempts_per_step = 5, lmbd_max = 1e+10, lmbd_min = 1e-10):
    """Minimize a scalar function of one or more variables using the
    Levenberg-Marquardt method.

    This variant uses an "exact" Levenberg-Marquardt routine based on 
    Cholesky factorization of the explicit Hessian matrix.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    disp = int(disp)
    xtol = x0.numel() * xtol
    if max_iter is None:
        max_iter = x0.numel() * 200

    # Construct scalar objective function
    sf = ScalarFunction(fun, x0.shape, hess=True)
    closure = sf.closure

    # initial settings
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    f, g, _, hess = closure(x)
    if disp > 1:
        print('initial fval: %0.4f' % f)
    if return_all:
        allvecs = [x]
    nfail = 0
    n_iter = 0
    lmbd = 1e-3

    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ==================================================
        #  Compute a search direction d by solving
        #          (H_f(x) + lambda * diag(H_f(x))) d = - J_f(x)
        #  with the true Hessian
        # ===================================================
        
        attempts = 0
        
        while True:
            
            attempts += 1
            
            if attempts > attempts_per_step:
                break
            
            hess_reg = hess + lmbd * torch.diag(hess.diag())
            # Compute search direction with Cholesky solve
            L, info = torch.linalg.cholesky_ex(hess_reg)
    
            if info == 0:
                d = torch.cholesky_solve(g.neg().unsqueeze(1), L).squeeze(1)
            else:
                nfail += 1
                d = torch.linalg.solve(hess_reg, g.neg())
    
            # =====================================================
            #  Perform variable (and regularization) update
            # =====================================================
    
            x = x + d  
            f_new = fun(x)
            
            if f_new >= f:             
                #Increase Levenber parameter unless it's already at its maximum
                lmbd = max(10 * lmbd, lmbd_max)
                #Overwrite the unregularized Hessian
                hess = hess_reg

            else:
                break
        
        # ===================================
        #  Re-evaluate func/Jacobian/Hessian
        # ===================================
        
        f, g, _, hess = closure(x)
        
        lmbd /= 10

        if disp > 1:
            print('iter %3d - fval: %0.4f - info: %d' % (n_iter, f, info))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)

        # ==========================
        #  check for convergence
        # ==========================

        if d.norm(p=normp) <= xtol:
            warnflag = 0
            msg = _status_message['success']
            break

        if not f.isfinite():
            warnflag = 3
            msg = _status_message['nan']
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
                            hess=hess.view(2 * x0.shape),
                            status=warnflag, success=(warnflag==0),
                            message=msg, nit=n_iter, nfev=sf.nfev, nfail=nfail)
    if return_all:
        result['allvecs'] = allvecs
    return result

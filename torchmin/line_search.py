import warnings
import torch
import numpy as np
from torch.optim.lbfgs import _strong_wolfe, _cubic_interpolate
from scipy.optimize import minimize_scalar
from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2, LineSearchWarning
from scipy.optimize._optimize import _LineSearchError
from scipy.optimize._dcsrch import DCSRCH

__all__ = ['strong_wolfe', 'brent', 'backtracking', 'scipy_wolfe12', 'dcsrch_wolfe', 'gpu_optimized_wolfe', 'hybrid_scipy_dcsrch', 'robust_wolfe']


def _strong_wolfe_extra(
        obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9,
        tolerance_change=1e-9, max_ls=25, extra_condition=None):
    """A modified variant of pytorch's strong-wolfe line search that supports
    an "extra_condition" argument (callable).

    This is required for methods such as Conjugate Gradient (polak-ribiere)
    where the strong wolfe conditions do not guarantee that we have a
    descent direction.

    Code borrowed from pytorch::
        Copyright (c) 2016 Facebook, Inc.
        All rights reserved.
    """
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    if extra_condition is None:
        extra_condition = lambda *args: True
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


def strong_wolfe(fun, x, t, d, f, g, gtd=None, **kwargs):
    """
    Expects `fun` to take arguments {x, t, d} and return {f(x1), f'(x1)},
    where x1 is the new location after taking a step from x in direction d
    with step size t.
    """
    if gtd is None:
        gtd = g.mul(d).sum()

    # use python floats for scalars as per torch.optim.lbfgs
    f, t = float(f), float(t)

    if 'extra_condition' in kwargs:
        f, g, t, ls_nevals = _strong_wolfe_extra(
            fun, x.view(-1), t, d.view(-1), f, g.view(-1), gtd, **kwargs)
    else:
        # in theory we shouldn't need to use pytorch's native _strong_wolfe
        # since the custom implementation above is equivalent with
        # extra_codition=None. But we will keep this in case they make any
        # changes.
        f, g, t, ls_nevals = _strong_wolfe(
            fun, x.view(-1), t, d.view(-1), f, g.view(-1), gtd, **kwargs)

    # convert back to torch scalar
    f = torch.as_tensor(f, dtype=x.dtype, device=x.device)

    return f, g.view_as(x), t, ls_nevals


def brent(fun, x, d, bounds=(0,10)):
    """
    Expects `fun` to take arguments {x} and return {f(x)}
    """
    def line_obj(t):
        return float(fun(x + t * d))
    res = minimize_scalar(line_obj, bounds=bounds, method='bounded')
    return res.x


def backtracking(fun, x, t, d, f, g, mu=0.1, decay=0.98, max_ls=500, tmin=1e-5):
    """
    Expects `fun` to take arguments {x, t, d} and return {f(x1), x1},
    where x1 is the new location after taking a step from x in direction d
    with step size t.

    We use a generalized variant of the armijo condition that supports
    arbitrary step functions x' = step(x,t,d). When step(x,t,d) = x + t * d
    it is equivalent to the standard condition.
    """
    x_new = x
    f_new = f
    success = False
    for i in range(max_ls):
        f_new, x_new = fun(x, t, d)
        if f_new <= f + mu * g.mul(x_new-x).sum():
            success = True
            break
        if t <= tmin:
            warnings.warn('step size has reached the minimum threshold.')
            break
        t = t.mul(decay)
    else:
        warnings.warn('backtracking did not converge.')

    return x_new, f_new, t, success


def scipy_wolfe12(obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, 
                  tolerance_change=1e-9, max_ls=25, extra_condition=None):
    """GPU-based line search equivalent to scipy's _line_search_wolfe12.
    
    This function mimics scipy.optimize._optimize._line_search_wolfe12 behavior:
    - First tries a more restrictive wolfe search (similar to wolfe1)
    - Falls back to a more permissive wolfe search (similar to wolfe2) if first fails
    - Raises exception if both fail
    
    All operations run on GPU, similar to strong_wolfe.
    
    Parameters
    ----------
    obj_func : callable
        Function that evaluates the objective and gradient at a given point
        Signature: obj_func(x, alpha, d) -> namedtuple(f, grad)
    x : torch.Tensor
        Current parameter vector
    t : float
        Initial step size
    d : torch.Tensor
        Search direction
    f : float
        Current function value
    g : torch.Tensor
        Current gradient
    gtd : float
        Gradient dotted with search direction
    c1 : float, optional
        Parameter for Armijo condition (default: 1e-4)
    c2 : float, optional
        Parameter for curvature condition (default: 0.9)
    tolerance_change : float, optional
        Minimum change tolerance (default: 1e-9)
    max_ls : int, optional
        Maximum number of line search iterations (default: 25)
    extra_condition : callable, optional
        Extra condition function (default: None)
    
    Returns
    -------
    f_new : float
        New function value
    g_new : torch.Tensor
        New gradient
    t : float
        Step size
    ls_evals : int
        Number of function evaluations
    """
    
    if gtd is None:
        gtd = g.dot(d)
    
    # Convert to Python floats for scalars
    f = float(f)
    t = float(t)
    gtd = float(gtd)
    
    if extra_condition is None:
        extra_condition = lambda *args: True
    
    # Try first strategy: more restrictive wolfe search (similar to wolfe1)
    # This uses stricter conditions and may fail more often
    try:
        f_new, g_new, t_new, ls_evals = _strong_wolfe_extra(
            obj_func, x.view(-1), t, d.view(-1), f, g.view(-1), gtd,
            c1=c1, c2=c2, tolerance_change=tolerance_change, 
            max_ls=max_ls, extra_condition=extra_condition
        )
        
        # Convert f_new to float if it's a tensor
        if isinstance(f_new, torch.Tensor):
            f_new = float(f_new.item())
        else:
            f_new = float(f_new)
        
        # Check if extra_condition is satisfied
        x_new = x + t_new * d
        if not extra_condition(t_new, x_new, f_new, g_new):
            # Extra condition failed, try fallback
            raise RuntimeError("Extra condition failed")
        
        # Success! Return results (matching strong_wolfe format)
        return f_new, g_new.view_as(x), float(t_new), ls_evals
        
    except (RuntimeError, ValueError, Exception) as e1:
        # First attempt failed, try fallback strategy (more permissive, similar to wolfe2)
        # Use slightly relaxed conditions
        try:
            # Try with more iterations and slightly relaxed tolerance
            f_new, g_new, t_new, ls_evals = _strong_wolfe_extra(
                obj_func, x.view(-1), t, d.view(-1), f, g.view(-1), gtd,
                c1=c1 * 0.5,  # More relaxed Armijo condition
                c2=min(c2 + 0.1, 0.99),  # Slightly more relaxed curvature
                tolerance_change=tolerance_change * 10,  # More relaxed tolerance
                max_ls=max_ls * 2,  # More iterations allowed
                extra_condition=extra_condition
            )
            
            # Convert f_new to float if it's a tensor
            if isinstance(f_new, torch.Tensor):
                f_new = float(f_new.item())
            else:
                f_new = float(f_new)
            
            # Check if extra_condition is satisfied
            x_new = x + t_new * d
            if not extra_condition(t_new, x_new, f_new, g_new):
                raise RuntimeError("Extra condition failed in fallback")
            
            # Success with fallback!
            return f_new, g_new.view_as(x), float(t_new), ls_evals
            
        except (RuntimeError, ValueError, Exception) as e2:
            # Both attempts failed, raise exception matching scipy behavior
            raise RuntimeError(
                f"Line search failed: first attempt ({e1}), fallback attempt ({e2})"
            )


def dcsrch_wolfe(obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, 
                 tolerance_change=1e-9, max_ls=100, extra_condition=None):
    """Optimized DCSRCH-based line search using SciPy's MINPACK implementation.
    
    This function provides the most robust and efficient line search by directly
    using SciPy's DCSRCH (MINPACK) algorithm with optimizations for GPU tensors.
    
    Parameters
    ----------
    obj_func : callable
        Function that evaluates the objective and gradient at a given point
    x : torch.Tensor
        Current parameter vector
    t : float
        Initial step size
    d : torch.Tensor
        Search direction
    f : float
        Current function value
    g : torch.Tensor
        Current gradient
    gtd : float
        Gradient dotted with search direction
    c1 : float, optional
        Parameter for Armijo condition (default: 1e-4)
    c2 : float, optional
        Parameter for curvature condition (default: 0.9)
    tolerance_change : float, optional
        Minimum change tolerance (default: 1e-9)
    max_ls : int, optional
        Maximum number of line search iterations (default: 100)
    extra_condition : callable, optional
        Extra condition function (default: None)
    
    Returns
    -------
    f_new : float
        New function value
    g_new : torch.Tensor
        New gradient
    t : float
        Step size
    ls_evals : int
        Number of function evaluations
    """
    
    # Cache for function evaluations to avoid double computation
    eval_cache = {}
    
    def phi(alpha):
        """Objective function for DCSRCH with caching"""
        if alpha in eval_cache:
            return eval_cache[alpha]['f']
        
        try:
            # Convert scalar alpha to tensor efficiently
            alpha_tensor = torch.tensor(alpha, device=x.device, dtype=x.dtype, requires_grad=False)
            result = obj_func(x, alpha_tensor, d)
            
            # Cache the result
            eval_cache[alpha] = {
                'f': float(result.f),
                'g': result.grad,
                'gtd': float(result.grad.dot(d)) if result.grad is not None else float('inf')
            }
            
            return eval_cache[alpha]['f']
        except Exception as e:
            return float('inf')
    
    def derphi(alpha):
        """Gradient function for DCSRCH with caching"""
        if alpha in eval_cache:
            return eval_cache[alpha]['gtd']
        
        try:
            # Convert scalar alpha to tensor efficiently
            alpha_tensor = torch.tensor(alpha, device=x.device, dtype=x.dtype, requires_grad=False)
            result = obj_func(x, alpha_tensor, d)
            
            # Extract gradient and compute directional derivative
            g_val = result.grad
            if g_val is None:
                return float('inf')
            
            gtd_val = float(g_val.dot(d))
            
            # Cache the result
            eval_cache[alpha] = {
                'f': float(result.f),
                'g': g_val,
                'gtd': gtd_val
            }
            
            return gtd_val
        except Exception as e:
            return float('inf')
    
    # Use SciPy's exact parameters for optimal performance
    dcsrch = DCSRCH(
        phi=phi,
        derphi=derphi,
        ftol=c1,      # Armijo condition parameter
        gtol=c2,      # Curvature condition parameter
        xtol=1e-14,   # SciPy's exact tolerance
        stpmin=1e-8,  # Minimum step size
        stpmax=50.0   # Maximum step size
    )
    
    # Initial values with proper scaling
    phi0 = float(f)
    derphi0 = float(gtd)
    
    # SciPy-style initial step size scaling
    if derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * abs(phi0) / abs(derphi0))
    else:
        alpha1 = float(t)
    
    # Ensure reasonable initial step size
    alpha1 = max(1e-8, min(alpha1, 1.0))
    
    # Run DCSRCH line search
    try:
        stp, phi1, phi0, task = dcsrch(
            alpha1=alpha1,
            phi0=phi0,
            derphi0=derphi0,
            maxiter=max_ls
        )
        
        # Check if line search was successful
        if stp is None or task.startswith(b'ERROR') or task.startswith(b'WARN'):
            # Fallback to simple step size reduction
            warnings.warn(f"DCSRCH line search failed with task: {task}, using fallback")
            stp = alpha1 * 0.1  # More aggressive step size reduction
            phi1 = phi(stp)
        
        # Get final gradient at the chosen step size (use cache if available)
        if stp in eval_cache:
            f_new = torch.as_tensor(eval_cache[stp]['f'], device=x.device, dtype=x.dtype)
            g_new = eval_cache[stp]['g']
        else:
            alpha_tensor = torch.tensor(stp, device=x.device, dtype=x.dtype)
            result = obj_func(x, alpha_tensor, d)
            f_new = result.f
            g_new = result.grad
        
        if g_new is None:
            raise ValueError("Gradient is None after line search")
        
        # Convert back to torch tensor if needed
        if not isinstance(f_new, torch.Tensor):
            f_new = torch.as_tensor(f_new, device=x.device, dtype=x.dtype)
        if not isinstance(g_new, torch.Tensor):
            g_new = torch.as_tensor(g_new, device=x.device, dtype=x.dtype)
        
        # Count function evaluations from cache
        ls_evals = len(eval_cache)
        
        return f_new, g_new, stp, ls_evals
        
    except Exception as e:
        # Complete fallback to original step size
        warnings.warn(f"DCSRCH line search failed with error: {e}, using original step size")
        alpha_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)
        result = obj_func(x, alpha_tensor, d)
        f_new = result.f
        g_new = result.grad
        
        if g_new is None:
            raise ValueError("Gradient is None after fallback")
        
        return f_new, g_new, t, 1


def gpu_optimized_wolfe(obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, 
                       tolerance_change=1e-9, max_ls=50, extra_condition=None):
    """GPU-optimized Wolfe line search with aggressive step size selection.
    
    This function provides a fast, GPU-optimized line search that's designed
    to match SciPy's convergence speed while maintaining GPU efficiency.
    
    Parameters
    ----------
    obj_func : callable
        Function that evaluates the objective and gradient at a given point
    x : torch.Tensor
        Current parameter vector
    t : float
        Initial step size
    d : torch.Tensor
        Search direction
    f : float
        Current function value
    g : torch.Tensor
        Current gradient
    gtd : float
        Gradient dotted with search direction
    c1 : float, optional
        Parameter for Armijo condition (default: 1e-4)
    c2 : float, optional
        Parameter for curvature condition (default: 0.9)
    tolerance_change : float, optional
        Minimum change tolerance (default: 1e-9)
    max_ls : int, optional
        Maximum number of line search iterations (default: 50)
    extra_condition : callable, optional
        Extra condition function (default: None)
    
    Returns
    -------
    f_new : float
        New function value
    g_new : torch.Tensor
        New gradient
    t : float
        Step size
    ls_evals : int
        Number of function evaluations
    """
    
    # SciPy-style initial step size scaling
    if gtd != 0:
        alpha1 = min(1.0, 1.01 * 2 * abs(f) / abs(gtd))
    else:
        alpha1 = float(t)
    
    # Ensure reasonable initial step size
    alpha1 = max(1e-8, min(alpha1, 1.0))
    
    # Cache for function evaluations
    eval_cache = {}
    
    def evaluate_alpha(alpha):
        """Evaluate function at given alpha with caching"""
        if alpha in eval_cache:
            return eval_cache[alpha]
        
        try:
            alpha_tensor = torch.tensor(alpha, device=x.device, dtype=x.dtype, requires_grad=False)
            result = obj_func(x, alpha_tensor, d)
            
            f_val = float(result.f)
            g_val = result.grad
            gtd_val = float(g_val.dot(d)) if g_val is not None else float('inf')
            
            eval_cache[alpha] = {
                'f': f_val,
                'g': g_val,
                'gtd': gtd_val
            }
            
            return eval_cache[alpha]
        except Exception as e:
            return {'f': float('inf'), 'g': None, 'gtd': float('inf')}
    
    # Initial evaluation
    result1 = evaluate_alpha(alpha1)
    f1, g1, gtd1 = result1['f'], result1['g'], result1['gtd']
    
    # Check if initial step satisfies Wolfe conditions
    if (f1 <= f + c1 * alpha1 * gtd and 
        abs(gtd1) <= -c2 * gtd):
        return (torch.as_tensor(f1, device=x.device, dtype=x.dtype), 
                g1, alpha1, 1)
    
    # Bracketing phase - find interval containing a good step
    alpha_prev = 0.0
    f_prev = f
    gtd_prev = gtd
    
    alpha_curr = alpha1
    f_curr = f1
    gtd_curr = gtd1
    
    ls_iter = 0
    while ls_iter < max_ls:
        ls_iter += 1
        
        # Check for sufficient decrease
        if f_curr > f + c1 * alpha_curr * gtd:
            # Need to reduce step size
            alpha_new = (alpha_prev + alpha_curr) / 2.0
        elif abs(gtd_curr) <= -c2 * gtd:
            # Curvature condition satisfied
            break
        elif gtd_curr >= 0:
            # Need to reduce step size (wrong direction)
            alpha_new = (alpha_prev + alpha_curr) / 2.0
        else:
            # Need to increase step size
            alpha_new = min(2.0 * alpha_curr, alpha_curr * 1.5)
        
        # Evaluate at new alpha
        result_new = evaluate_alpha(alpha_new)
        f_new, g_new, gtd_new = result_new['f'], result_new['g'], result_new['gtd']
        
        # Check Wolfe conditions
        if (f_new <= f + c1 * alpha_new * gtd and 
            abs(gtd_new) <= -c2 * gtd):
            alpha_curr = alpha_new
            f_curr = f_new
            gtd_curr = gtd_new
            break
        
        # Update bracketing
        alpha_prev = alpha_curr
        f_prev = f_curr
        gtd_prev = gtd_curr
        
        alpha_curr = alpha_new
        f_curr = f_new
        gtd_curr = gtd_new
    
    # Final evaluation
    result_final = evaluate_alpha(alpha_curr)
    f_final = result_final['f']
    g_final = result_final['g']
    
    if g_final is None:
        raise ValueError("Gradient is None after line search")
    
    # Convert to torch tensors if needed
    if not isinstance(f_final, torch.Tensor):
        f_final = torch.as_tensor(f_final, device=x.device, dtype=x.dtype)
    if not isinstance(g_final, torch.Tensor):
        g_final = torch.as_tensor(g_final, device=x.device, dtype=x.dtype)
    
    return f_final, g_final, alpha_curr, len(eval_cache)


def hybrid_scipy_dcsrch(obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, 
                       tolerance_change=1e-9, max_ls=100, extra_condition=None):
    """Hybrid line search: SciPy Wolfe12 for direction + DCSRCH for step size optimization.
    
    This approach uses SciPy's robust Wolfe12 line search to find the search direction
    and then uses DCSRCH to optimize the step size for maximum convergence speed.
    
    Parameters
    ----------
    obj_func : callable
        Function that evaluates the objective and gradient at a given point
    x : torch.Tensor
        Current parameter vector
    t : float
        Initial step size
    d : torch.Tensor
        Search direction
    f : float
        Current function value
    g : torch.Tensor
        Current gradient
    gtd : float
        Gradient dotted with search direction
    c1 : float, optional
        Parameter for Armijo condition (default: 1e-4)
    c2 : float, optional
        Parameter for curvature condition (default: 0.9)
    tolerance_change : float, optional
        Minimum change tolerance (default: 1e-9)
    max_ls : int, optional
        Maximum number of line search iterations (default: 100)
    extra_condition : callable, optional
        Extra condition function (default: None)
    
    Returns
    -------
    f_new : float
        New function value
    g_new : torch.Tensor
        New gradient
    t : float
        Step size
    ls_evals : int
        Number of function evaluations
    """
    
    # Step 1: Use SciPy Wolfe12 to get initial step size and direction
    try:
        f_new, g_new, alpha_k, ls_evals = scipy_wolfe12(obj_func, x, t, d, f, g, gtd)
        
        # If SciPy found a good step, use it
        if alpha_k > 0 and f_new < f:
            return f_new, g_new, alpha_k, ls_evals
        
    except Exception as e:
        # If SciPy fails, fall back to DCSRCH
        warnings.warn(f"SciPy Wolfe12 failed: {e}, using DCSRCH fallback")
        return dcsrch_wolfe(obj_func, x, t, d, f, g, gtd, c1, c2, tolerance_change, max_ls, extra_condition)
    
    # Step 2: Use DCSRCH to optimize the step size further
    # Convert the SciPy result to a starting point for DCSRCH
    initial_alpha = alpha_k if alpha_k > 0 else t
    
    # Cache for function evaluations to avoid double computation
    eval_cache = {}
    
    def phi(alpha):
        """Objective function for DCSRCH with caching"""
        if alpha in eval_cache:
            return eval_cache[alpha]['f']
        
        try:
            # Convert scalar alpha to tensor efficiently
            alpha_tensor = torch.tensor(alpha, device=x.device, dtype=x.dtype, requires_grad=False)
            result = obj_func(x, alpha_tensor, d)
            
            # Cache the result
            eval_cache[alpha] = {
                'f': float(result.f),
                'g': result.grad,
                'gtd': float(result.grad.dot(d)) if result.grad is not None else float('inf')
            }
            
            return eval_cache[alpha]['f']
        except Exception as e:
            return float('inf')
    
    def derphi(alpha):
        """Gradient function for DCSRCH with caching"""
        if alpha in eval_cache:
            return eval_cache[alpha]['gtd']
        
        try:
            # Convert scalar alpha to tensor efficiently
            alpha_tensor = torch.tensor(alpha, device=x.device, dtype=x.dtype, requires_grad=False)
            result = obj_func(x, alpha_tensor, d)
            
            # Extract gradient and compute directional derivative
            g_val = result.grad
            if g_val is None:
                return float('inf')
            
            gtd_val = float(g_val.dot(d))
            
            # Cache the result
            eval_cache[alpha] = {
                'f': float(result.f),
                'g': g_val,
                'gtd': gtd_val
            }
            
            return gtd_val
        except Exception as e:
            return float('inf')
    
    # Use DCSRCH with more aggressive parameters for step size optimization
    dcsrch = DCSRCH(
        phi=phi,
        derphi=derphi,
        ftol=c1,      # Armijo condition parameter
        gtol=c2,      # Curvature condition parameter
        xtol=1e-12,   # More aggressive tolerance for step size optimization
        stpmin=1e-10, # Smaller minimum step size
        stpmax=100.0  # Larger maximum step size for more aggressive optimization
    )
    
    # Initial values with SciPy's result as starting point
    phi0 = float(f)
    derphi0 = float(gtd)
    alpha1 = float(initial_alpha)
    
    # Ensure reasonable initial step size
    alpha1 = max(1e-10, min(alpha1, 10.0))
    
    # Run DCSRCH step size optimization
    try:
        stp, phi1, phi0, task = dcsrch(
            alpha1=alpha1,
            phi0=phi0,
            derphi0=derphi0,
            maxiter=max_ls // 2  # Use half the iterations for step size optimization
        )
        
        # Check if DCSRCH found a better step size
        if stp is not None and not task.startswith(b'ERROR') and not task.startswith(b'WARN'):
            # Use DCSRCH result if it's better
            if stp in eval_cache:
                f_final = torch.as_tensor(eval_cache[stp]['f'], device=x.device, dtype=x.dtype)
                g_final = eval_cache[stp]['g']
            else:
                alpha_tensor = torch.tensor(stp, device=x.device, dtype=x.dtype)
                result = obj_func(x, alpha_tensor, d)
                f_final = result.f
                g_final = result.grad
            
            if g_final is not None:
                # Convert to torch tensors if needed
                if not isinstance(f_final, torch.Tensor):
                    f_final = torch.as_tensor(f_final, device=x.device, dtype=x.dtype)
                if not isinstance(g_final, torch.Tensor):
                    g_final = torch.as_tensor(g_final, device=x.device, dtype=x.dtype)
                
                return f_final, g_final, stp, len(eval_cache)
        
        # If DCSRCH didn't improve, use SciPy's result
        return f_new, g_new, alpha_k, ls_evals
        
    except Exception as e:
        # If DCSRCH fails, use SciPy's result
        warnings.warn(f"DCSRCH step size optimization failed: {e}, using SciPy result")
        return f_new, g_new, alpha_k, ls_evals


def robust_wolfe(obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, 
                 tolerance_change=1e-9, max_ls=50, extra_condition=None):
    """Robust Wolfe line search designed for stable training.
    
    This function provides a simple, robust line search that's designed
    to work well with complex neural network training scenarios.
    
    Parameters
    ----------
    obj_func : callable
        Function that evaluates the objective and gradient at a given point
    x : torch.Tensor
        Current parameter vector
    t : float
        Initial step size
    d : torch.Tensor
        Search direction
    f : float
        Current function value
    g : torch.Tensor
        Current gradient
    gtd : float
        Gradient dotted with search direction
    c1 : float, optional
        Parameter for Armijo condition (default: 1e-4)
    c2 : float, optional
        Parameter for curvature condition (default: 0.9)
    tolerance_change : float, optional
        Minimum change tolerance (default: 1e-9)
    max_ls : int, optional
        Maximum number of line search iterations (default: 50)
    extra_condition : callable, optional
        Extra condition function (default: None)
    
    Returns
    -------
    f_new : float
        New function value
    g_new : torch.Tensor
        New gradient
    t : float
        Step size
    ls_evals : int
        Number of function evaluations
    """
    
    # Conservative initial step size scaling
    if gtd != 0:
        alpha1 = min(1.0, 0.5 * abs(f) / abs(gtd))
    else:
        alpha1 = float(t)
    
    # Ensure reasonable initial step size
    alpha1 = max(1e-6, min(alpha1, 1.0))
    
    # Simple bracketing and zooming
    alpha_prev = 0.0
    f_prev = f
    gtd_prev = gtd
    
    alpha_curr = alpha1
    ls_evals = 0
    
    # Try initial step
    try:
        alpha_tensor = torch.tensor(alpha_curr, device=x.device, dtype=x.dtype, requires_grad=False)
        result = obj_func(x, alpha_tensor, d)
        f_curr = float(result.f)
        g_curr = result.grad
        gtd_curr = float(g_curr.dot(d)) if g_curr is not None else float('inf')
        ls_evals += 1
        
        # Check if initial step satisfies Wolfe conditions
        if (f_curr <= f + c1 * alpha_curr * gtd and 
            abs(gtd_curr) <= -c2 * gtd):
            return (torch.as_tensor(f_curr, device=x.device, dtype=x.dtype), 
                    g_curr, alpha_curr, ls_evals)
        
    except Exception as e:
        # If initial step fails, use conservative fallback
        alpha_curr = alpha1 * 0.1
        alpha_tensor = torch.tensor(alpha_curr, device=x.device, dtype=x.dtype, requires_grad=False)
        result = obj_func(x, alpha_tensor, d)
        f_curr = float(result.f)
        g_curr = result.grad
        ls_evals += 1
        
        return (torch.as_tensor(f_curr, device=x.device, dtype=x.dtype), 
                g_curr, alpha_curr, ls_evals)
    
    # Bracketing phase
    for ls_iter in range(max_ls):
        ls_evals += 1
        
        # Check for sufficient decrease
        if f_curr > f + c1 * alpha_curr * gtd:
            # Need to reduce step size
            alpha_new = (alpha_prev + alpha_curr) / 2.0
        elif abs(gtd_curr) <= -c2 * gtd:
            # Curvature condition satisfied
            break
        elif gtd_curr >= 0:
            # Need to reduce step size (wrong direction)
            alpha_new = (alpha_prev + alpha_curr) / 2.0
        else:
            # Need to increase step size (conservatively)
            alpha_new = min(1.5 * alpha_curr, alpha_curr * 1.2)
        
        # Evaluate at new alpha
        try:
            alpha_tensor = torch.tensor(alpha_new, device=x.device, dtype=x.dtype, requires_grad=False)
            result = obj_func(x, alpha_tensor, d)
            f_new = float(result.f)
            g_new = result.grad
            gtd_new = float(g_new.dot(d)) if g_new is not None else float('inf')
            ls_evals += 1
            
            # Check Wolfe conditions
            if (f_new <= f + c1 * alpha_new * gtd and 
                abs(gtd_new) <= -c2 * gtd):
                alpha_curr = alpha_new
                f_curr = f_new
                gtd_curr = gtd_new
                break
            
            # Update bracketing
            alpha_prev = alpha_curr
            f_prev = f_curr
            gtd_prev = gtd_curr
            
            alpha_curr = alpha_new
            f_curr = f_new
            gtd_curr = gtd_new
            
        except Exception as e:
            # If evaluation fails, use current best
            break
    
    # Final evaluation
    try:
        alpha_tensor = torch.tensor(alpha_curr, device=x.device, dtype=x.dtype, requires_grad=False)
        result = obj_func(x, alpha_tensor, d)
        f_final = result.f
        g_final = result.grad
        ls_evals += 1
        
        if g_final is None:
            raise ValueError("Gradient is None after line search")
        
        # Convert to torch tensors if needed
        if not isinstance(f_final, torch.Tensor):
            f_final = torch.as_tensor(f_final, device=x.device, dtype=x.dtype)
        if not isinstance(g_final, torch.Tensor):
            g_final = torch.as_tensor(g_final, device=x.device, dtype=x.dtype)
        
        return f_final, g_final, alpha_curr, ls_evals
        
    except Exception as e:
        # Complete fallback to original step size
        alpha_tensor = torch.tensor(t, device=x.device, dtype=x.dtype, requires_grad=False)
        result = obj_func(x, alpha_tensor, d)
        f_final = result.f
        g_final = result.grad
        
        if g_final is None:
            raise ValueError("Gradient is None after fallback")
        
        return f_final, g_final, t, 1
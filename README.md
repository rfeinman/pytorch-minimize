# PyTorch Minimize

This library contains a collection of utilities for minimizing scalar functions of one or more variables in PyTorch. 
It is inspired heavily by SciPy's `optimize` module and MATLAB's [Optimization Toolbox](https://www.mathworks.com/products/optimization.html). 
Unlike SciPy and MATLAB, jacobian and hessian functions need not be provided to pytorch-minimize solvers, and numerical approximations are never used.
At the moment, only unconstrained minimization is supported.

Author: Reuben Feinman

__At a glance:__

```python
import torch
from fmin import minimize

def rosen(x):
    return torch.sum(100*(x[..., 1:] - x[..., :-1]**2)**2 
                     + (1 - x[..., :-1])**2)

# initial point
x0 = torch.tensor([1., 8.])

# BFGS
result = minimize(rosen, x0, method='bfgs')

# Newton Conjugate Gradient
result = minimize(rosen, x0, method='newton-cg')

# Newton Exact
result = minimize(rosen, x0, method='newton-exact')
```

__Solvers:__ BFGS, L-BFGS, Newton Conjugate Gradient (CG), Newton Exact

__Examples:__ See the [Rosenbrock minimization notebook](https://github.com/rfeinman/pytorch-minimize/blob/master/examples/rosen_minimize.ipynb) for a demonstration of function minimization with a handful of different algorithms.

## Motivation
Although PyTorch offers many routines for stochastic optimization, utilities for deterministic optimization are scarce; only L-BFGS is included in the `optim` package, and it's modified for mini-batch training.

MATLAB and SciPy are industry standards for deterministic optimization. 
These libraries have a comprehensive set of routines; however, automatic differentiation is not supported.* 
Therefore, the user must specify 1st- and 2nd-order gradients explicitly (if they are known) or use finite-difference approximations.

The motivation for this library is to offer a set of tools for deterministic optimization with analytical gradients via PyTorch's autograd.

_

*MATLAB offers minimal autograd support via the Deep Learning Toolbox, but the integration is not seamless: data must be converted to "dlarray" structures, and only a [subset of functions](https://www.mathworks.com/help/deeplearning/ug/list-of-functions-with-dlarray-support.html) are supported.

## Minimization Routines

1. __BFGS/L-BFGS.__ BFGS is a cannonical quasi-Newton method for unconstrained optimization. I've implemented both the standard BFGS and the "limited memory" L-BFGS. For smaller scale problems where memory is not a concern, BFGS should be significantly faster than L-BFGS (especially on CUDA) since it avoids Python for loops and instead uses pure torch.
   
2. __Newton Conjugate Gradient (CG).__ The Newton-Raphson method is a staple of unconstrained optimization. Although computing full Hessian matrices with PyTorch's reverse-mode automatic differentiation can be costly, computing Hessian-vector products is cheap (and it saves a lot of memory). The Conjugate Gradient (CG) variant of Newton's method is an effective solution for unconstrained minimization with Hessian-vector products. I've implemented a lightweight NewtonCG minimizer that uses HVP for the linear inverse sub-problems.

3. __Newton Exact.__ In some cases, we may prefer a more precise variant of the Newton-Raphson method at the cost of additional complexity. I've also implemented an "exact" variant of Newton's method that computes the full Hessian matrix and uses Cholesky factorization for linear inverse sub-problems. When Cholesky fails--i.e. the Hessian is not positive definite--the solver resorts to LU factorization. This is not the safest resolution for ill-conditioned Hessians and it may be modified in the future.

## Examples

The [Rosenbrock minimization tutorial](https://github.com/rfeinman/pytorch-minimize/blob/master/examples/rosen_minimize.ipynb) demonstrates how to use pytorch-minimize to find the minimum of a scalar-valued function of multiple variables using various optimization strategies.

In addition, the [SciPy benchmark](https://github.com/rfeinman/pytorch-minimize/blob/master/examples/scipy_benchmark.py) provides a comparison of pytorch-minimize solvers to their analogous solvers from the `scipy.optimize` library. 
For those transitioning from scipy, this script will help get a feel for the design of the current library. 
Unlike scipy, jacobian and hessian functions need not be provided to pytorch-minimize solvers, and numerical approximations are never used.

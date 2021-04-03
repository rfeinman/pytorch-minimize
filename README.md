# PyTorch Minimize

This library contains a collection of utilities for minimizing scalar functions of one or more variables in PyTorch. It is inspired heavily by SciPy's `optimize` module and MATLAB's [Optimization Toolbox](https://www.mathworks.com/products/optimization.html).

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
```

__Solvers:__ BFGS, L-BFGS, Newton Conjugate Gradient (CG), Newton Exact

__Examples:__ See `/examples/rosen_minimize.ipynb` for a demonstration of function minimization with a handful of different algorithms.

## Motivation
Although PyTorch offers many routines for stochastic optimization, utilities for deterministic optimization are scarce; only L-BFGS is included in the `optim` package, and it's modified for mini-batch training.

MATLAB and SciPy are the industry standards for deterministic optimization. 
These libraries have a comprehensive set of routines; however, automatic differentiation is not supported. 
Therefore, the user must specify 1st- and 2nd-order gradients explicitly (if they are known) or use finite-difference approximations.

The motivation for this library is to offer a set of tools for deterministic optimization with analytical gradients via PyTorch's autograd.

## Minimization Routines

1. __BFGS.__ BFGS is a cannonical quasi-Newton method for unconstrained optimization. I've implemented both the standard BFGS and the "limited memory" L-BFGS, accessed from the same function with argument `low_mem=True`. For smaller scale problems where memory is not a major concern, BFGS should be significantly faster than L-BFGS since it avoids for loops and uses pure torch matrix math. This is especially the case on CUDA.
   
2. __Newton Conjugate Gradient (CG).__ Due to the use of reverse-mode automatic differentiation, computing explicit Hessian matrices with PyTorch's autograd is not very efficient. However, computing Hessian-vector products is fast (and it also saves a lot of memory). The Conjugate Gradient (CG) variant of the Newton-Raphson algorithm is an effective solution for unconstrained minimization with Hessian-vector products. I've implemented a lightweight NewtonCG minimizer that uses HVP for the linear inverse sub-problems.

3. __Newton Exact.__ In some cases, we may prefer a more precise variant of the Newton-Raphson method at the cost of additional complexity. I've included an "exact" variant of Newton's method that computes the full Hessian matrix and uses Cholesky factorization for linear inverse sub-problems. When Cholesky fails--i.e. the Hessian is not positive definite--the solver resorts to LU factorization. This is not the safest resolution for indefinite Hessians and may be modified in the future.


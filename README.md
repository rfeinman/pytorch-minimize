# PyTorch Minimize

For the most up-to-date information on pytorch-minimize, see the docs site: [pytorch-minimize.readthedocs.io](https://pytorch-minimize.readthedocs.io/)

Pytorch-minimize represents a collection of utilities for minimizing multivariate functions in PyTorch. 
It is inspired heavily by SciPy's `optimize` module and MATLAB's [Optimization Toolbox](https://www.mathworks.com/products/optimization.html). 
Unlike SciPy and MATLAB, which use numerical approximations of function derivatives, pytorch-minimize uses _real_ first- and second-order derivatives, computed seamlessly behind the scenes with autograd.
Both CPU and CUDA are supported.

__Author__: Reuben Feinman

__At a glance:__

```python
import torch
from torchmin import minimize

def rosen(x):
    return torch.sum(100*(x[..., 1:] - x[..., :-1]**2)**2 
                     + (1 - x[..., :-1])**2)

# initial point
x0 = torch.tensor([1., 8.])

# Select from the following methods:
#  ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
#   'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']

# BFGS
result = minimize(rosen, x0, method='bfgs')

# Newton Conjugate Gradient
result = minimize(rosen, x0, method='newton-cg')

# Newton Exact
result = minimize(rosen, x0, method='newton-exact')
```

__Solvers:__ BFGS, L-BFGS, Conjugate Gradient (CG), Newton Conjugate Gradient (NCG), Newton Exact, Dogleg, Trust-Region Exact, Trust-Region NCG, Trust-Region GLTR (Krylov)

__Examples:__ See the [Rosenbrock minimization notebook](https://github.com/rfeinman/pytorch-minimize/blob/master/examples/rosen_minimize.ipynb) for a demonstration of function minimization with a handful of different algorithms.

__Install with pip:__

    pip install pytorch-minimize

__Install from source:__

    git clone https://github.com/rfeinman/pytorch-minimize.git
    cd pytorch-minimize
    pip install -e .

## Motivation
Although PyTorch offers many routines for stochastic optimization, utilities for deterministic optimization are scarce; only L-BFGS is included in the `optim` package, and it's modified for mini-batch training.

MATLAB and SciPy are industry standards for deterministic optimization. 
These libraries have a comprehensive set of routines; however, automatic differentiation is not supported.* 
Therefore, the user must provide explicit 1st- and 2nd-order gradients (if they are known) or use finite-difference approximations.

The motivation for pytorch-minimize is to offer a set of tools for deterministic optimization with automatic gradients and GPU acceleration.

__

*MATLAB offers minimal autograd support via the Deep Learning Toolbox, but the integration is not seamless: data must be converted to "dlarray" structures, and only a [subset of functions](https://www.mathworks.com/help/deeplearning/ug/list-of-functions-with-dlarray-support.html) are supported.
Furthermore, derivatives must still be constructed and provided as function handles. 
Pytorch-minimize uses autograd to compute derivatives behind the scenes, so all you provide is an objective function.

## Library

The pytorch-minimize library includes solvers for general-purpose function minimization (unconstrained & constrained), as well as for nonlinear least squares problems.

### 1. Unconstrained Minimizers

The following solvers are available for _unconstrained_ minimization:

- __BFGS/L-BFGS.__ BFGS is a cannonical quasi-Newton method for unconstrained optimization. I've implemented both the standard BFGS and the "limited memory" L-BFGS. For smaller scale problems where memory is not a concern, BFGS should be significantly faster than L-BFGS (especially on CUDA) since it avoids Python for loops and instead uses pure torch.

- __Conjugate Gradient (CG).__ The conjugate gradient algorithm is a generalization of linear conjugate gradient to nonlinear optimization problems. Pytorch-minimize includes an implementation of the Polak-Ribiére CG algorithm described in Nocedal & Wright (2006) chapter 5.2.
   
- __Newton Conjugate Gradient (NCG).__ The Newton-Raphson method is a staple of unconstrained optimization. Although computing full Hessian matrices with PyTorch's reverse-mode automatic differentiation can be costly, computing Hessian-vector products is cheap, and it also saves a lot of memory. The Conjugate Gradient (CG) variant of Newton's method is an effective solution for unconstrained minimization with Hessian-vector products. I've implemented a lightweight NewtonCG minimizer that uses HVP for the linear inverse sub-problems.

- __Newton Exact.__ In some cases, we may prefer a more precise variant of the Newton-Raphson method at the cost of additional complexity. I've also implemented an "exact" variant of Newton's method that computes the full Hessian matrix and uses Cholesky factorization for linear inverse sub-problems. When Cholesky fails--i.e. the Hessian is not positive definite--the solver resorts to one of two options as specified by the user: 1) steepest descent direction (default), or 2) solve the inverse hessian with LU factorization.

- __Trust-Region Newton Conjugate Gradient.__ Description coming soon.

- __Trust-Region Newton Generalized Lanczos (Krylov).__ Description coming soon.

- __Trust-Region Exact.__ Description coming soon.

- __Dogleg.__ Description coming soon.

To access the unconstrained minimizer interface, use the following import statement:

    from torchmin import minimize

Use the argument `method` to specify which of the afformentioned solvers should be applied.

### 2. Constrained Minimizers

The following solvers are available for _constrained_ minimization:

- __Trust-Region Constrained Algorithm.__ Pytorch-minimize includes a single constrained minimization routine based on SciPy's 'trust-constr' method. The algorithm accepts generalized nonlinear constraints and variable boundries via the "constr" and "bounds" arguments. For equality constrained problems, it is an implementation of the Byrd-Omojokun Trust-Region SQP method. When inequality constraints are imposed, the trust-region interior point method is used. NOTE: The current trust-region constrained minimizer is not a custom implementation, but rather a wrapper for SciPy's `optimize.minimize` routine. It uses autograd behind the scenes to build jacobian & hessian callables before invoking scipy. Inputs and objectivs should use torch tensors like other pytorch-minimize routines. CUDA is supported but not recommended; data will be moved back-and-forth between GPU/CPU. 
   
To access the constrained minimizer interface, use the following import statement:

    from torchmin import minimize_constr

### 3. Nonlinear Least Squares

The library also includes specialized solvers for nonlinear least squares problems. 
These solvers revolve around the Gauss-Newton method, a modification of Newton's method tailored to the lstsq setting. 
The least squares interface can be imported as follows:

    from torchmin import least_squares

The least_squares function is heavily motivated by scipy's `optimize.least_squares`. 
Much of the scipy code was borrowed directly (all rights reserved) and ported from numpy to torch. 
Rather than have the user provide a jacobian function, in the new interface, jacobian-vector products are computed behind the scenes with autograd. 
At the moment, only the Trust Region Reflective ("trf") method is implemented, and bounds are not yet supported.

## Examples

The [Rosenbrock minimization tutorial](https://github.com/rfeinman/pytorch-minimize/blob/master/examples/rosen_minimize.ipynb) demonstrates how to use pytorch-minimize to find the minimum of a scalar-valued function of multiple variables using various optimization strategies.

In addition, the [SciPy benchmark](https://github.com/rfeinman/pytorch-minimize/blob/master/examples/scipy_benchmark.py) provides a comparison of pytorch-minimize solvers to their analogous solvers from the `scipy.optimize` library. 
For those transitioning from scipy, this script will help get a feel for the design of the current library. 
Unlike scipy, jacobian and hessian functions need not be provided to pytorch-minimize solvers, and numerical approximations are never used.

For constrained optimization, the [adversarial examples tutorial](https://github.com/rfeinman/pytorch-minimize/blob/master/examples/constrained_optimization_adversarial_examples.ipynb) demonstrates how to use the trust-region constrained routine to generate an optimal adversarial perturbation given a constraint on the perturbation norm.

## Optimizer API

As an alternative to the functional API, pytorch-minimize also includes an "optimizer" API based on the `torch.optim.Optimizer` class. 
To access the optimizer class, import as follows:

    from torchmin import Minimizer

## Citing this work

If you use pytorch-minimize for academic research, you may cite the library as follows:

```
@misc{Feinman2021,
  author = {Feinman, Reuben},
  title = {Pytorch-minimize: a library for numerical optimization with autograd},
  publisher = {GitHub},
  year = {2021},
  url = {https://github.com/rfeinman/pytorch-minimize},
}
```

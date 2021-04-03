# PyTorch Minimize

This library contains a collection of utilities for minimizing scalar functions of one or more variables in Pytorch. It is inspired heavily by SciPy's `optimize` module and MATLAB's [Optimization Toolbox](https://www.mathworks.com/products/optimization.html).

## Motivation
Although PyTorch offers many routines for stochastic optimization, utilities for deterministic optimization are scarce; only L-BFGS is included in the `optim` package, and it's modified for mini-batch training.

MATLAB and SciPy are the industry standards for deterministic optimization. 
These libraries have a comprehensive set of routines; however, automatic differentiation is not supported. 
Therefore, the user must either specify Jacobian and Hessian functions explicitly (if they are known) or use finite-difference approximations.

The motivation for this library is to offer a set of tools for deterministic optimization with analytical gradients via PyTorch's autograd.


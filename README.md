# PyTorch Minimize

This library contains a collection of utilities for minimizing scalar functions of one or more variables in Pytorch. It is inspired heavily by scipy's `optimize` module and MATLAB's [Optimization Toolbox](https://www.mathworks.com/products/optimization.html).

## Motivation
Although PyTorch offers many routines for stochastic optimization, utilities for deterministic optimization are scarce; only L-BFGS is included in the `optim` package, and it's a modified variant designed for mini-batch training.

MATLAB and scipy are the industry standards for deterministic optimization. 
These libraries have a comprehensive set of routines; however, automatic differentiation is not supported. 
Therefore, the user must either specify Jacobian and Hessian functions explicitly or use finite-difference approximations. 
This limits the applications to new areas of research that autograd has enabled.

The motivation for this library is to develop a new set of tools for deterministic optimization build around PyTorch's autograd mechanics.


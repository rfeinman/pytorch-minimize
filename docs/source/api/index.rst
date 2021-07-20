=================
API Documentation
=================

.. currentmodule:: torchmin


Functional API
==============

The functional API provides an interface similar to those of SciPy's :mod:`optimize` module and MATLAB's ``fminunc``/``fmincon`` routines. Parameters are provided as a single torch Tensor, and an :class:`OptimizeResult` instance is returned that includes the optimized parameter value as well as other useful information (e.g. final function value, parameter gradient, etc.).

There are 3 core utilities in the functional API, designed for 3 unique
numerical optimization problems.


**Unconstrained minimization**

.. autosummary::
    :toctree: generated

    minimize

The :func:`minimize` function is a general utility for *unconstrained* minimization. It implements a number of different routines based on Newton and Quasi-Newton methods for numerical optimization. The following methods are supported, accessed via the `method` argument:

.. toctree::

    minimize-bfgs
    minimize-lbfgs
    minimize-cg
    minimize-newton-cg
    minimize-newton-exact
    minimize-dogleg
    minimize-trust-ncg
    minimize-trust-exact
    minimize-trust-krylov


**Constrained minimization**

.. autosummary::
    :toctree: generated

    minimize_constr

The :func:`minimize_constr` function is a general utility for *constrained* minimization. Algorithms for constrained minimization use Newton and Quasi-Newton methods on the KKT conditions of the constrained optimization problem.

.. note::
    The :func:`minimize_constr` function is currently in early beta. Unlike :func:`minimize`--which uses custom, pure PyTorch backend--the constrained solver is a wrapper for SciPy's 'trust-constr' minimization method. CUDA tensors are supported, but CUDA will only be used for function and gradient evaluation, with the remaining solver computations performed on CPU (with numpy arrays).


**Nonlinear least-squares**

.. autosummary::
    :toctree: generated

    least_squares

The :func:`least_squares` function is a specialized utility for nonlinear least-squares minimization problems. Algorithms for least-squares revolve around the Gauss-Newton method, a modification of Newton's method tailored to residual sum-of-squares (RSS) optimization. The following methods are currently supported:

- Trust-region reflective
- Dogleg - COMING SOON
- Gauss-Newton line search - COMING SOON



Optimizer API
==============

The optimizer API provides an alternative interface based on PyTorch's :mod:`optim` module. This interface follows the schematic of PyTorch optimizers and will be familiar to those migrating from torch.

.. autosummary::
    :toctree: generated

    Minimizer
    Minimizer.step

The :class:`Minimizer` class inherits from :class:`torch.optim.Optimizer` and constructs an object that holds the state of the provided variables. Unlike the functional API, which expects parameters to be a single Tensor, parameters can be passed to :class:`Minimizer` as iterables of Tensors. The class serves as a wrapper for :func:`torchmin.minimize()` and can use any of its methods (selected via the `method` argument) to perform unconstrained minimization.

.. autosummary::
    :toctree: generated

    ScipyMinimizer
    ScipyMinimizer.step

Although the :class:`Minimizer` class will be sufficient for most problems where torch optimizers would be used, it does not support constraints. Another optimizer is provided, :class:`ScipyMinimizer`, which supports parameter bounds and linear/nonlinear constraint functions. This optimizer is a wrapper for :func:`scipy.optimize.minimize`. When using bound constraints, `bounds` are passed as iterables with same length as `params`, i.e. one bound specification per parameter Tensor.

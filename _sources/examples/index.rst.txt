Examples
=========

The examples site is in active development. Check back soon for more complete examples of how to use pytorch-minimize.

Unconstrained minimization
---------------------------

.. code-block:: python

    from torchmin import minimize
    from torchmin.benchmarks import rosen

    # initial point
    x0 = torch.randn(100, device='cpu')

    # BFGS
    result = minimize(rosen, x0, method='bfgs')

    # Newton Conjugate Gradient
    result = minimize(rosen, x0, method='newton-cg')

Constrained minimization
---------------------------

For constrained optimization, the `adversarial examples tutorial <https://github.com/rfeinman/pytorch-minimize/blob/master/examples/constrained_optimization_adversarial_examples.ipynb>`_ demonstrates how to use trust-region constrained optimization to generate an optimal adversarial perturbation given a constraint on the perturbation norm.

Nonlinear least-squares
---------------------------

Coming soon.


Scipy benchmark
---------------------------

The `SciPy benchmark <https://github.com/rfeinman/pytorch-minimize/blob/master/examples/scipy_benchmark.py>`_ provides a comparison of pytorch-minimize solvers to their analogous solvers from the :mod:`scipy.optimize` module.
For those transitioning from scipy, this script will help get a feel for the design of the current library.
Unlike scipy, jacobian and hessian functions need not be provided to pytorch-minimize solvers, and numerical approximations are never used.


Minimizer (optimizer API)
---------------------------

Another way to use the optimization tools from pytorch-minimize is via :class:`torchmin.Minimizer`, a pytorch Optimizer class. For a demo on how to use the Minimizer class, see the `MNIST classifier <https://github.com/rfeinman/pytorch-minimize/blob/master/examples/train_mnist_Minimizer.py>`_ tutorial.
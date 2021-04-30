Tutorial
=========

This page is a placeholder. The tutorial site will be coming soon.

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

To do.

Nonlinear least-Squares
---------------------------

To do.
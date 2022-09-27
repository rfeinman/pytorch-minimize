Pytorch-minimize
================

Pytorch-minimize is a library for numerical optimization with automatic differentiation and GPU acceleration. It implements a number of canonical techniques for deterministic (or "full-batch") optimization not offered in the :mod:`torch.optim` module. The library is inspired heavily by SciPy's :mod:`optimize` module and MATLAB's `Optimization Toolbox <https://www.mathworks.com/products/optimization.html>`_. Unlike SciPy and MATLAB, which use numerical approximations of derivatives that are slow and often inaccurate, pytorch-minimize uses *real* first- and second-order derivatives, computed seamlessly behind the scenes with autograd. Both CPU and CUDA are supported.

:Author: Reuben Feinman
:Version: 0.0.1

Pytorch-minimize is currently in Beta; expect the API to change before a first official release. Some of the source code was taken directly from SciPy and ported to PyTorch. As such, here is their copyright notice:

    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers. All rights reserved.


Table of Contents
=================

.. toctree::
    :maxdepth: 2

    install

.. toctree::
    :maxdepth: 2

    user_guide/index

.. toctree::
    :maxdepth: 2

    api/index

.. toctree::
    :maxdepth: 2

    examples/index

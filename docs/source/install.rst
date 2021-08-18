Install
===========

To install pytorch-minimize, first clone the repository to a location of your choice::

    git clone https://github.com/rfeinman/pytorch-minimize.git

Next, cd to the root directory and install the package with pip::

    cd pytorch-minimize
    pip install -e .

Once these steps have been completed, you should be able to make inports such as ``from torchmin import minimize`` from any location on your machine.

**PyTorch requirement**

This library uses latest features from the actively-developed :mod:`torch.linalg` module. For maximum performance, users should install pytorch>=1.9, as it includes some new items not available in prior releases (e.g. `torch.linalg.cholesky_ex <https://pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html>`_). Pytorch-minimize will automatically use these features when available.

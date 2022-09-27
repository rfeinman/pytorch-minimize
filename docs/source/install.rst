Install
===========

To install pytorch-minimize, users may either 1) install the official PyPI release via pip, or 2) install a *bleeding edge* distribution from source.


**Install via pip (official PyPI release)**::

    pip install pytorch-minimize

**Install from source (bleeding edge)**::

    # clone the latest master to any location
    git clone https://github.com/rfeinman/pytorch-minimize.git

    # cd to the root directory and install the package with pip
    cd pytorch-minimize
    pip install -e .


**PyTorch requirement**

This library uses latest features from the actively-developed :mod:`torch.linalg` module. For maximum performance, users should install pytorch>=1.9, as it includes some new items not available in prior releases (e.g. `torch.linalg.cholesky_ex <https://pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html>`_). Pytorch-minimize will automatically use these features when available.

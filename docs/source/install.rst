Install
===========

Pytorch-minimize has not yet been published on PyPi, therefore, it cannot be install using pip. For now, the suggested method is to download the source code repository and add its root directory to your `PYTHONPATH`. A full list of requirements is provided in `requirements.txt <https://github.com/rfeinman/pytorch-minimize/blob/master/requirements.txt>`_.

First, run the following command to clone the repository to a folder of your choice::

    git clone https://github.com/rfeinman/pytorch-minimize.git

Next, add the selected folder path to your `PYTHONPATH` environment variable as follows::

    export PYTHONPATH="/path/to/pytorch-minimize:$PYTHONPATH"

Once these steps have been completed, you should be able to make inports such as ``from torchmin import minimize`` from any location on your machine.
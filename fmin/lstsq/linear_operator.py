import torch
from scipy.sparse.linalg import LinearOperator


def _is_cuda(device):
    if device is None:
        return False
    if isinstance(device, str):
        return 'cuda' in device
    if isinstance(device, torch.device):
        return 'cuda' in device.type
    raise RuntimeError('invalid device encountered.')


def wrap_numpy(fun, device, dtype):
    def new_fun(inp):
        if _is_cuda(device):
            inp = torch.tensor(inp, device=device, dtype=dtype)
        else:
            # potentially avoid data copy
            inp = torch.from_numpy(inp).to(dtype)
        out = fun(inp)
        return out.data.cpu().numpy()
    return new_fun


class TorchLinearOperator(object):
    """Linear operator defined in terms of user-specified operations."""
    def __init__(self, shape, matvec, rmatvec=None, device=None, dtype=None):
        self.shape = shape
        self._matvec = matvec
        self._rmatvec = rmatvec
        self.device = device
        self.dtype = dtype

    def matvec(self, x):
        return self._matvec(x)

    def rmatvec(self, x):
        if self._rmatvec is None:
            raise NotImplementedError("rmatvec is not defined")
        return self._rmatvec(x)

    def matmat(self, X):
        return torch.hstack([self.matvec(col).view(-1,1) for col in X.T])

    mv = matvec
    rmv = rmatvec
    matmul = matmat

    def scipy(self):
        matvec = wrap_numpy(self._matvec, self.device, self.dtype)
        if self._rmatvec is not None:
            rmatvec = wrap_numpy(self._rmatvec, self.device, self.dtype)
        else:
            rmatvec = None
        return LinearOperator(self.shape, matvec=matvec, rmatvec=rmatvec)
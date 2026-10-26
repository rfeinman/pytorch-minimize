import pytest
import torch

from torchmin.lstsq.lsmr import lsmr


@pytest.mark.parametrize('xtrue', [torch.zeros(10), torch.ones(10)])
def test_lsmr_trivial_system(xtrue):
    # Regression test for #20: these used to return all NaNs.
    A = torch.eye(10)
    b = A @ xtrue
    x, _ = lsmr(A, b)
    torch.testing.assert_close(x, xtrue)


def test_lsmr_x0_is_solution():
    A = torch.eye(5)
    x0 = torch.arange(5.)
    x, itn = lsmr(A, A @ x0, x0=x0)
    assert itn == 0
    torch.testing.assert_close(x, x0)

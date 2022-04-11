import pytest
import torch

from torchmin import minimize

torch.manual_seed(42)
N = 100
D = 7
M = 5
X = torch.randn(N, D)
Y = torch.randn(N, M)
trueB = torch.linalg.inv(X.T @ X) @ X.T @ Y
all_methods = [
    'bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact',
    'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']


@pytest.mark.parametrize('method', all_methods)
def test_minimize(method):
    """Test least-squares problem on unconstrained minimizers."""
    B0 = torch.zeros(D, M)

    def leastsquares_obj(B):
        return torch.sum((Y - X @ B) ** 2)

    result = minimize(leastsquares_obj, B0, method=method)
    torch.testing.assert_close(trueB, result.x, rtol=1e-4, atol=1e-4)

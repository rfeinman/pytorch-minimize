import pytest
import torch
from scipy.optimize import Bounds

from torchmin import minimize, minimize_constr
from torchmin.benchmarks import rosen


@pytest.mark.parametrize(
    'method',
    ['l-bfgs-b', 'trust-constr'],
)
def test_equivalent_bounds(method):
    x0 = torch.tensor([-1.0, 1.5])

    def minimize_with_bounds(bounds):
        return minimize_constr(
            rosen,
            x0,
            method=method,
            bounds=bounds,
            tol=1e-6,
        )

    def assert_equivalent(src_result, tgt_result):
        return torch.testing.assert_close(
            src_result.x,
            tgt_result.x,
            rtol=1e-5,
            atol=1e-3,
            msg=f"Solution {src_result.x} not close to expected {tgt_result.x}"
        )

    result_0 = minimize_with_bounds(
        bounds=(torch.tensor([-2.0, -2.0]), torch.tensor([2.0, 2.0]))
    )

    equivalent_bounds_to_test = [
        ([-2.0, -2.0], [2.0, 2.0]),
        (-2.0, 2.0),
        Bounds(-2.0, 2.0),
    ]
    for bounds in equivalent_bounds_to_test:
        result = minimize_with_bounds(bounds)
        assert_equivalent(result, result_0)
        print(f'Test passed with bounds: {bounds}')


def test_invalid_bounds():
    x0 = torch.tensor([-1.0, 1.5])

    invalid_bounds_to_test = [
        (torch.tensor([-2.0]), torch.tensor([2.0, 2.0])),
        (-2.0,),
        torch.tensor([-2.0, -2.0, 2.0, 2.0]),
    ]

    for bounds in invalid_bounds_to_test:
        with pytest.raises(Exception):
            result = minimize_constr(
                rosen,
                x0,
                method='l-bfgs-b',
                bounds=bounds,
            )


# TODO: remove this block
if __name__ == '__main__':
    test_equivalent_bounds(method='l-bfgs-b')
    test_invalid_bounds()

"""Simple test for L-BFGS-B implementation.

TODO: rewrite this with `pytest`.
"""
import torch
from torchmin.lbfgsb import _minimize_constr_lbfgsb


def test_simple_quadratic():
    """Test L-BFGS-B on a simple bounded quadratic problem.

    Minimize: f(x) = (x1 - 2)^2 + (x2 - 1)^2
    Subject to: 0 <= x1 <= 1.5, 0 <= x2 <= 2

    The unconstrained minimum is at (2, 1), but x1 is constrained,
    so the optimal solution should be at (1.5, 1).
    """
    print("Test 1: Simple quadratic with bounds")
    print("=" * 50)

    def fun(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2

    x0 = torch.tensor([0.5, 0.5])
    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([1.5, 2.0])

    result = _minimize_constr_lbfgsb(
        fun, x0, bounds=(lb, ub),
        gtol=1e-6, ftol=1e-9, disp=1
    )

    print(f"\nOptimal x: {result.x}")
    print(f"Optimal f: {result.fun:.6f}")
    print(f"Expected x: [1.5, 1.0]")
    print(f"Expected f: 0.25")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    print()

    # Check if close to expected solution
    expected_x = torch.tensor([1.5, 1.0])
    expected_f = 0.25

    assert torch.allclose(result.x, expected_x, atol=1e-4), \
        f"Solution {result.x} not close to expected {expected_x}"
    assert abs(result.fun - expected_f) < 1e-4, \
        f"Function value {result.fun} not close to expected {expected_f}"

    print("✓ Test 1 passed!\n")


def test_rosenbrock():
    """Test L-BFGS-B on Rosenbrock function with bounds.

    Minimize: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Subject to: -2 <= x <= 2, -2 <= y <= 2

    The unconstrained minimum is at (1, 1).
    """
    print("Test 2: Rosenbrock function with bounds")
    print("=" * 50)

    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    x0 = torch.tensor([-1.0, 1.5])
    lb = torch.tensor([-2.0, -2.0])
    ub = torch.tensor([2.0, 2.0])

    result = _minimize_constr_lbfgsb(
        rosenbrock, x0, bounds=(lb, ub),
        gtol=1e-6, ftol=1e-9, max_iter=100, disp=1
    )

    print(f"\nOptimal x: {result.x}")
    print(f"Optimal f: {result.fun:.6e}")
    print(f"Expected x: [1.0, 1.0]")
    print(f"Expected f: 0.0")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    print()

    # Check if close to expected solution
    expected_x = torch.tensor([1.0, 1.0])

    assert torch.allclose(result.x, expected_x, atol=1e-3), \
        f"Solution {result.x} not close to expected {expected_x}"
    assert result.fun < 1e-6, \
        f"Function value {result.fun} not close to 0"

    print("✓ Test 2 passed!\n")


def test_active_constraints():
    """Test with multiple active constraints.

    Minimize: f(x) = sum(x_i^2)
    Subject to: x_i >= 1 for all i

    The solution should be all ones (on the boundary).
    """
    print("Test 3: Multiple active constraints")
    print("=" * 50)

    def fun(x):
        return (x**2).sum()

    n = 5
    x0 = torch.ones(n) * 2.0
    lb = torch.ones(n)
    ub = torch.ones(n) * 10.0

    result = _minimize_constr_lbfgsb(
        fun, x0, bounds=(lb, ub),
        gtol=1e-6, ftol=1e-9, disp=1
    )

    print(f"\nOptimal x: {result.x}")
    print(f"Optimal f: {result.fun:.6f}")
    print(f"Expected x: [1.0, 1.0, 1.0, 1.0, 1.0]")
    print(f"Expected f: 5.0")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    print()

    expected_x = torch.ones(n)
    expected_f = float(n)

    assert torch.allclose(result.x, expected_x, atol=1e-4), \
        f"Solution {result.x} not close to expected {expected_x}"
    assert abs(result.fun - expected_f) < 1e-4, \
        f"Function value {result.fun} not close to expected {expected_f}"

    print("✓ Test 3 passed!\n")


if __name__ == "__main__":
    test_simple_quadratic()
    test_rosenbrock()
    test_active_constraints()
    print("=" * 50)
    print("All tests passed! ✓")

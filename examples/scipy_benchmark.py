"""
A comparison of pytorch-minimize solvers to the analogous solvers from
scipy.optimize.

Pytorch-minimize uses autograd to compute 1st- and 2nd-order derivatives
implicitly, therefore derivative functions need not be provided or known.
In contrast, scipy.optimize requires that they be provided, or else it will
use imprecise numerical approximations. For fair comparison I am providing
derivative functions to scipy.optimize in this script. In general, however,
we will not have access to these functions, so applications of scipy.optimize
are far more limited.

"""
import torch
from torchmin import minimize
from torchmin.benchmarks import rosen
from scipy import optimize

# Many scipy optimizers convert the data to double-precision, so
# we will use double precision in torch for a fair comparison
torch.set_default_dtype(torch.float64)


def print_header(title, num_breaks=1):
    print('\n'*num_breaks + '='*50)
    print(' '*20 + title)
    print('='*50 + '\n')


def main():
    torch.manual_seed(991)
    x0 = torch.randn(100)
    x0_np = x0.numpy()

    print('\ninitial loss: %0.4f\n' % rosen(x0))


    # ---- BFGS ----
    print_header('BFGS')

    print('-'*19 + ' pytorch ' + '-'*19)
    res = minimize(rosen, x0, method='bfgs', tol=1e-5, disp=True)

    print('\n' + '-'*20 + ' scipy ' + '-'*20)
    res = optimize.minimize(
        optimize.rosen, x0_np,
        method='bfgs',
        jac=optimize.rosen_der,
        tol=1e-5,
        options=dict(disp=True)
    )


    # ---- Newton CG ----
    print_header('Newton-CG')

    print('-'*19 + ' pytorch ' + '-'*19)
    res = minimize(rosen, x0, method='newton-cg', tol=1e-5, disp=True)

    print('\n' + '-'*20 + ' scipy ' + '-'*20)
    res = optimize.minimize(
        optimize.rosen, x0_np,
        method='newton-cg',
        jac=optimize.rosen_der,
        hessp=optimize.rosen_hess_prod,
        tol=1e-5,
        options=dict(disp=True)
    )


    # ---- Newton Exact ----
    # NOTE: Scipy does not have a precise analogue to "newton-exact," but they
    # have something very close called "trust-exact." Like newton-exact,
    # trust-exact also uses Cholesky factorization of the explicit Hessian
    # matrix. However, whereas newton-exact first computes the newton direction
    # and then uses line search to determine a step size, trust-exact first
    # specifies a step size boundary and then solves for the optimal newton
    # step within this boundary (a constrained optimization problem).

    print_header('Newton-Exact')

    print('-'*19 + ' pytorch ' + '-'*19)
    res = minimize(rosen, x0, method='newton-exact', tol=1e-5, disp=True)

    print('\n' + '-'*20 + ' scipy ' + '-'*20)
    res = optimize.minimize(
        optimize.rosen, x0_np,
        method='trust-exact',
        jac=optimize.rosen_der,
        hess=optimize.rosen_hess,
        options=dict(gtol=1e-5, disp=True)
    )

    print()


if __name__ == '__main__':
    main()
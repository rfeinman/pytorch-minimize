"""
Generic interface for nonlinear least-squares minimization.
"""
from warnings import warn
import numbers
import torch

from .trf import trf
from .common import EPS, in_bounds, make_strictly_feasible

__all__ = ['least_squares']


TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied."
}


def prepare_bounds(bounds, x0):
    n = x0.shape[0]
    def process(b):
        if isinstance(b, numbers.Number):
            return x0.new_full((n,), b)
        elif isinstance(b, torch.Tensor):
            if b.dim() == 0:
                return x0.new_full((n,), b)
            return b
        else:
            raise ValueError

    lb, ub = [process(b) for b in bounds]

    return lb, ub


def check_tolerance(ftol, xtol, gtol, method):
    def check(tol, name):
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn("Setting `{}` below the machine epsilon ({:.2e}) effectively "
                 "disables the corresponding termination condition."
                 .format(name, EPS))
        return tol

    ftol = check(ftol, "ftol")
    xtol = check(xtol, "xtol")
    gtol = check(gtol, "gtol")

    if method == "lm" and (ftol < EPS or xtol < EPS or gtol < EPS):
        raise ValueError("All tolerances must be higher than machine epsilon "
                         "({:.2e}) for method 'lm'.".format(EPS))
    elif ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError("At least one of the tolerances must be higher than "
                         "machine epsilon ({:.2e}).".format(EPS))

    return ftol, xtol, gtol


def check_x_scale(x_scale, x0):
    if isinstance(x_scale, str) and x_scale == 'jac':
        return x_scale
    try:
        x_scale = torch.as_tensor(x_scale)
        valid = x_scale.isfinite().all() and x_scale.gt(0).all()
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with "
                         "positive numbers.")

    if x_scale.dim() == 0:
        x_scale = x0.new_full(x0.shape, x_scale)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale


def least_squares(
        fun, x0, bounds=None, method='trf', ftol=1e-8, xtol=1e-8,
        gtol=1e-8, x_scale=1.0, tr_solver='lsmr', tr_options=None,
        max_nfev=None, verbose=0):
    r"""Solve a nonlinear least-squares problem with bounds on the variables.

    Given the residual function
    :math:`f: \mathcal{R}^n \rightarrow \mathcal{R}^m`, `least_squares`
    finds a local minimum of the residual sum-of-squares (RSS) objective:

    .. math::
        x^* = \underset{x}{\operatorname{arg\,min\,}}
        \frac{1}{2} ||f(x)||_2^2 \quad \text{subject to} \quad lb \leq x \leq ub

    The solution is found using variants of the Gauss-Newton method, a
    modification of Newton's method tailored to RSS problems.

    Parameters
    ----------
    fun : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x)``. The argument ``x`` passed to this
        function is a Tensor of shape (n,) (never a scalar, even for n=1).
        It must allocate and return a 1-D Tensor of shape (m,) or a scalar.
    x0 : Tensor or float
        Initial guess on independent variables, with shape (n,). If
        float, it will be treated as a 1-D Tensor with one element.
    bounds : 2-tuple of Tensor, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each Tensor must match the size of `x0` or be a scalar, in the latter
        case a bound will be the same for all variables. Use ``inf`` with
        an appropriate sign to disable bounds on all or some variables.
    method : str, optional
        Algorithm to perform minimization. Default is 'trf'.

            * 'trf' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            * 'dogbox' : COMING SOON. dogleg algorithm with rectangular trust regions,
              typical use case is small problems with bounds. Not recommended
              for problems with rank-deficient Jacobian.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. The
        optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step. If None, the termination by this
        condition is disabled. Default is 1e-8.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Termination occurs when ``norm(dx) < xtol * (xtol + norm(x))``.
        If None, the termination by this condition is disabled. Default is 1e-8.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
        The exact condition depends on `method` used:

            * For 'trf' : ``norm(g_scaled, ord=inf) < gtol``, where
              ``g_scaled`` is the value of the gradient scaled to account for
              the presence of the bounds [STIR]_.
            * For 'dogbox' : ``norm(g_free, ord=inf) < gtol``, where
              ``g_free`` is the gradient with respect to the variables which
              are not in the optimal state on the boundary.
    x_scale : Tensor or 'jac', optional
        Characteristic scale of each variable. Setting `x_scale` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting `x_scale` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to 'jac', the scale is iteratively updated using the
        inverse norms of the columns of the Jacobian matrix (as described in
        [JJMore]_).
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination.
        Defaults to 100 * n.
    tr_solver : str, optional
        Method for solving trust-region subproblems.

            * 'exact' is suitable for not very large problems with dense
              Jacobian matrices. The computational complexity per iteration is
              comparable to a singular value decomposition of the Jacobian
              matrix.
            * 'lsmr' is suitable for problems with sparse and large Jacobian
              matrices. It uses an iterative procedure for finding a solution
              of a linear least-squares problem and only requires matrix-vector
              product evaluations.
    tr_options : dict, optional
        Keyword options passed to trust-region solver.

            * ``tr_solver='exact'``: `tr_options` are ignored.
            * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
              Additionally,  ``method='trf'`` supports  'regularize' option
              (bool, default is True), which adds a regularization term to the
              normal equation, which improves convergence if the Jacobian is
              rank-deficient [Byrd]_ (eq. 3.4).
    verbose : int, optional
        Level of algorithm's verbosity.

            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    References
    ----------
    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
              and Conjugate Gradient Method for Large-Scale Bound-Constrained
              Minimization Problems," SIAM Journal on Scientific Computing,
              Vol. 21, Number 1, pp 1-23, 1999.
    .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate
              solution of the trust region problem by minimization over
              two-dimensional subspaces", Math. Programming, 40, pp. 247-263,
              1988.
    .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation
                and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
                Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.

    """
    if tr_options is None:
        tr_options = {}

    if method not in ['trf', 'dogbox']:
        raise ValueError("`method` must be 'trf' or 'dogbox'.")

    if tr_solver not in ['exact', 'lsmr', 'cgls']:
        raise ValueError("`tr_solver` must be one of {'exact', 'lsmr', 'cgls'}.")

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if bounds is None:
        bounds = (-float('inf'), float('inf'))
    elif not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
        raise ValueError("`bounds` must be a tuple/list with 2 elements.")

    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")

    # initial point
    x0 = torch.atleast_1d(x0)
    if torch.is_complex(x0):
        raise ValueError("`x0` must be real.")
    elif x0.dim() > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    # bounds
    lb, ub = prepare_bounds(bounds, x0)
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")
    elif torch.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")
    elif not in_bounds(x0, lb, ub):
        raise ValueError("`x0` is infeasible.")

    # x_scale
    x_scale = check_x_scale(x_scale, x0)

    # tolerance
    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)

    if method == 'trf':
        x0 = make_strictly_feasible(x0, lb, ub)

    def fun_wrapped(x):
        return torch.atleast_1d(fun(x))

    # check function
    f0 = fun_wrapped(x0)
    if f0.dim() != 1:
        raise ValueError("`fun` must return at most 1-d array_like. "
                         "f0.shape: {0}".format(f0.shape))
    elif not f0.isfinite().all():
        raise ValueError("Residuals are not finite in the initial point.")

    initial_cost = 0.5 * f0.dot(f0)

    if isinstance(x_scale, str) and x_scale == 'jac':
        raise ValueError("x_scale='jac' can't be used when `jac` "
                         "returns LinearOperator.")

    if method == 'trf':
        result = trf(fun_wrapped, x0, f0, lb, ub, ftol, xtol, gtol,
                     max_nfev, x_scale, tr_solver, tr_options.copy(), verbose)
    elif method == 'dogbox':
        raise NotImplementedError("'dogbox' method not yet implemented")
        # if tr_solver == 'lsmr' and 'regularize' in tr_options:
        #     warn("The keyword 'regularize' in `tr_options` is not relevant "
        #          "for 'dogbox' method.")
        #     tr_options = tr_options.copy()
        #     del tr_options['regularize']
        # result = dogbox(fun_wrapped, x0, f0, lb, ub, ftol, xtol, gtol,
        #                 max_nfev, x_scale, tr_solver, tr_options, verbose)
    else:
        raise ValueError("`method` must be 'trf' or 'dogbox'.")

    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0

    if verbose >= 1:
        print(result.message)
        print("Function evaluations {0}, initial cost {1:.4e}, final cost "
              "{2:.4e}, first-order optimality {3:.2e}."
              .format(result.nfev, initial_cost, result.cost,
                      result.optimality))

    return result
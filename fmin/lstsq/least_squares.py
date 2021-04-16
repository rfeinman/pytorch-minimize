"""
Generic interface for nonlinear least-squares minimization.
"""
from warnings import warn
import numbers
import torch

from .trf import trf
from .common import EPS, in_bounds, make_strictly_feasible


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
        max_nfev=None, verbose=0, args=(), kwargs=None):
    """Solve a nonlinear least-squares problem with bounds on the variables.

    Given the residuals f(x) (an m-D real function of n real
    variables) and the loss function rho(s) (a scalar function), `least_squares`
    finds a local minimum of the cost function F(x)::
        minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
        subject to lb <= x <= ub
    The purpose of the loss function rho(s) is to reduce the influence of
    outliers on the solution.

    Parameters
    ----------
    fun : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
        respect to its first argument. The argument ``x`` passed to this
        function is an ndarray of shape (n,) (never a scalar, even for n=1).
        It must allocate and return a 1-D array_like of shape (m,) or a scalar.
        If the argument ``x`` is complex or the function ``fun`` returns
        complex residuals, it must be wrapped in a real function of real
        arguments, as shown at the end of the Examples section.
    x0 : array_like with shape (n,) or float
        Initial guess on independent variables. If float, it will be treated
        as a 1-D array with one element.
    jac : {'2-point', '3-point', 'cs', callable}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]). The keywords select a finite difference scheme for numerical
        estimation. The scheme '3-point' is more accurate, but requires
        twice as many operations as '2-point' (default). The scheme 'cs'
        uses complex steps, and while potentially the most accurate, it is
        applicable only when `fun` correctly handles complex inputs and
        can be analytically continued to the complex plane. Method 'lm'
        always uses the '2-point' scheme. If callable, it is used as
        ``jac(x, *args, **kwargs)`` and should return a good approximation
        (or the exact value) for the Jacobian as an array_like (np.atleast_2d
        is applied), a sparse matrix (csr_matrix preferred for performance) or
        a `scipy.sparse.linalg.LinearOperator`.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of `x0` or be a scalar, in the latter
        case a bound will be the same for all variables. Use ``np.inf`` with
        an appropriate sign to disable bounds on all or some variables.
    method : {'trf', 'dogbox', 'lm'}, optional
        Algorithm to perform minimization.
            * 'trf' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            * 'dogbox' : dogleg algorithm with rectangular trust regions,
              typical use case is small problems with bounds. Not recommended
              for problems with rank-deficient Jacobian.
            * 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
              Doesn't handle bounds and sparse Jacobians. Usually the most
              efficient method for small unconstrained problems.
        Default is 'trf'. See Notes for more information.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step.
        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. The exact condition depends on the `method` used:
            * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``.
            * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
              a trust-region radius and ``xs`` is the value of ``x``
              scaled according to `x_scale` parameter (see below).
        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
        The exact condition depends on a `method` used:
            * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
              ``g_scaled`` is the value of the gradient scaled to account for
              the presence of the bounds [STIR]_.
            * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
              ``g_free`` is the gradient with respect to the variables which
              are not in the optimal state on the boundary.
            * For 'lm' : the maximum absolute value of the cosine of angles
              between columns of the Jacobian and the residual vector is less
              than `gtol`, or the residual vector is zero.
        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    x_scale : array_like or 'jac', optional
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
        If None (default), the value is chosen automatically:
            * For 'trf' and 'dogbox' : 100 * n.
            * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
              otherwise (because 'lm' counts function calls in Jacobian
              estimation).
    tr_solver : {None, 'exact', 'lsmr'}, optional
        Method for solving trust-region subproblems, relevant only for 'trf'
        and 'dogbox' methods.
            * 'exact' is suitable for not very large problems with dense
              Jacobian matrices. The computational complexity per iteration is
              comparable to a singular value decomposition of the Jacobian
              matrix.
            * 'lsmr' is suitable for problems with sparse and large Jacobian
              matrices. It uses the iterative procedure
              `scipy.sparse.linalg.lsmr` for finding a solution of a linear
              least-squares problem and only requires matrix-vector product
              evaluations.
        If None (default), the solver is chosen based on the type of Jacobian
        returned on the first iteration.
    tr_options : dict, optional
        Keyword options passed to trust-region solver.
            * ``tr_solver='exact'``: `tr_options` are ignored.
            * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
              Additionally,  ``method='trf'`` supports  'regularize' option
              (bool, default is True), which adds a regularization term to the
              normal equation, which improves convergence if the Jacobian is
              rank-deficient [Byrd]_ (eq. 3.4).
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations (not supported by 'lm'
              method).
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun` and `jac`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)`` and the same for
        `jac`.

    """
    if tr_options is None:
        tr_options = {}
    if kwargs is None:
        kwargs = {}

    if method not in ['trf', 'dogbox']:
        raise ValueError("`method` must be 'trf' or 'dogbox'.")

    if tr_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")
    elif tr_solver == 'exact':
        raise NotImplementedError("exact trust solver not currently supported.")

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
        return torch.atleast_1d(fun(x, *args, **kwargs))

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
                     max_nfev, x_scale, tr_options.copy(), verbose)
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
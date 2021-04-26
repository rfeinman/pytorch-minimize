"""
Trust-region optimization.

Code ported from SciPy to PyTorch

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.
"""
from abc import ABC, abstractmethod
import torch
from torch.linalg import norm
from scipy.optimize.optimize import OptimizeResult, _status_message

__all__ = []

status_messages = (
    _status_message['success'],
    _status_message['maxiter'],
    'A bad approximation caused failure to predict improvement.',
    'A linalg error occurred, such as a non-psd Hessian.',
)


class BaseQuadraticSubproblem(ABC):
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method.
    Values of the objective function, Jacobian and Hessian (if provided) at
    the current iterate ``x`` are evaluated on demand and then stored as
    attributes ``fun``, ``jac``, ``hess``.
    """

    def __init__(self, x, fun, jac, hess=None, hessp=None):
        self._x = x
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp
        # buffer for boundaries computation
        self._tab = x.new_empty(2)

    def __call__(self, p):
        return self.fun + self.jac.dot(p) + 0.5 * p.dot(self.hessp(p))

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        """Value of Jacobian of objective function at current iteration."""
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        """Value of Hessian of objective function at current iteration."""
        if self._h is None:
            self._h = self._hess(self._x)
        return self._h

    def hessp(self, p):
        if self._hessp is not None:
            return self._hessp(self._x, p)
        else:
            return self.hess.mv(p)

    @property
    def jac_mag(self):
        """Magnitude of jacobian of objective function at current iteration."""
        if self._g_mag is None:
            self._g_mag = norm(self.jac)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = d.dot(d)
        b = 2 * z.dot(d)
        c = z.dot(z) - trust_radius**2
        sqrt_discriminant = torch.sqrt(b*b - 4*a*c)

        # The following calculation is mathematically equivalent to:
        #   ta = (-b - sqrt_discriminant) / (2*a)
        #   tb = (-b + sqrt_discriminant) / (2*a)
        # but produces smaller round off errors.
        aux = b + torch.copysign(sqrt_discriminant, b)
        self._tab[0] = -aux / (2*a)
        self._tab[1] = -2*c / aux
        return self._tab.sort()[0]


    @abstractmethod
    def solve(self, trust_radius):
        pass


def _minimize_trust_region(fun, x0, jac=None, hess=None, hessp=None,
                           subproblem=None, initial_trust_radius=1.,
                           max_trust_radius=1000., eta=0.15, gtol=1e-4,
                           maxiter=None, disp=False, return_all=False,
                           callback=None):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """

    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region '
                         'methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is currently required for trust-region methods')
    if subproblem is None:
        raise ValueError('A subproblem solving strategy is required for '
                         'trust-region methods')
    if not (0 <= eta < 0.25):
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the '
                         'max trust radius')

    # force the initial guess into a nice format
    x0 = torch.as_tensor(x0).flatten()

    # limit the number of iterations
    if maxiter is None:
        maxiter = x0.numel() * 200

    # init the search status
    #warnflag = 0
    #k = 0

    # initialize the search
    trust_radius = torch.as_tensor(initial_trust_radius,
                                   dtype=x0.dtype, device=x0.device)
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)

    # search for the function min
    # do not even start if the gradient is small enough
    for k in range(maxiter):

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            p, hits_boundary = m.solve(trust_radius)
        except RuntimeError as exc:
            # TODO: catch general linalg error like np.linalg.linalg.LinAlgError
            if 'singular' in exc.args[0]:
                warnflag = 3
                break
            else:
                raise

        # calculate the predicted value at the proposed point
        predicted_value = m(p)

        # define the local approximation at the proposed point
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)

        # evaluate the ratio defined in equation (4.4)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        if rho < 0.25:
            trust_radius = trust_radius.mul(0.25)
        elif rho > 0.75 and hits_boundary:
            trust_radius = torch.clamp(2*trust_radius, max=max_trust_radius)

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed

        # append the best guess, call back, increment the iteration count
        if return_all:
            allvecs.append(x.clone())
        if callback is not None:
            callback(x.clone())

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            k += 1
            warnflag = 0
            break

    else:
        # maximum iterations reached
        k += 1
        warnflag = 1

    # print some stuff if requested
    if disp:
        msg = status_messages[warnflag]
        if warnflag != 0:
            msg = 'Warning: ' + msg
        print(msg)
        print("         Current function value: %f" % m.fun)
        print("         Iterations: %d" % k)
        # print("         Function evaluations: %d" % sf.nfev)
        # print("         Gradient evaluations: %d" % sf.ngev)
        # print("         Hessian evaluations: %d" % (sf.nhev + nhessp[0]))

    result = OptimizeResult(x=x, fun=m.fun, jac=m.jac,
                            success=(warnflag == 0), status=warnflag,
                            # nfev=sf.nfev, njev=sf.ngev, nhev=sf.nhev+nhessp[0],
                            nit=k, message=status_messages[warnflag])

    if hess is not None:
        result['hess'] = m.hess

    if return_all:
        result['allvecs'] = allvecs

    return result
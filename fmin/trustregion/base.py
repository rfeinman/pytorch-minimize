"""
Trust-region optimization.

Code ported from SciPy to PyTorch

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.
"""
from abc import ABC, abstractmethod
import torch
import torch.autograd as autograd
from torch._vmap_internals import _vmap
from torch.linalg import norm
from scipy.optimize.optimize import OptimizeResult, _status_message


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
    """
    def __init__(self, x, fun):
        self._x = x

        # compute function and jacobian value (with grad enabled)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = fun(x)
            g, = autograd.grad(f, x, create_graph=True)

        # build hessp function or compute hessian matrix
        if self.hess_prod:
            h = None
            hessp = lambda v: autograd.grad(g, x, v, retain_graph=True)[0]
        else:
            I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            h = _vmap(lambda v: autograd.grad(g, x, v)[0])(I)
            hessp = None

        self._f = f.detach()
        self._g = g.detach()
        self._h = h
        self._hessp = hessp
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None

        # buffer for boundaries computation
        self._tab = x.new_empty(2)

    def __call__(self, p):
        return self.fun + self.jac.dot(p) + 0.5 * p.dot(self.hessp(p))

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        return self._f

    @property
    def jac(self):
        """Value of Jacobian of objective function at current iteration."""
        return self._g

    @property
    def hess(self):
        """Value of Hessian of objective function at current iteration."""
        if self.hess_prod:
            raise Exception('class {} does not have '
                            'method `hess`'.format(type(self)))
        return self._h

    def hessp(self, p):
        if self.hess_prod:
            return self._hessp(p)
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

    @property
    @abstractmethod
    def hess_prod(self):
        """A property that must be set by every sub-class indicating whether
        to use full hessian matrix or hessian-vector products."""
        pass


def _minimize_trust_region(fun, x0, subproblem=None, initial_trust_radius=1.,
                           max_trust_radius=1000., eta=0.15, gtol=1e-4,
                           max_iter=None, disp=False, return_all=False,
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
        max_iter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """
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
    if max_iter is None:
        max_iter = x0.numel() * 200

    # init the search status
    warnflag = 1  # maximum iterations flag
    k = 0

    # initialize the search
    trust_radius = torch.as_tensor(initial_trust_radius,
                                   dtype=x0.dtype, device=x0.device)
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun)

    # search for the function min
    # do not even start if the gradient is small enough
    while k < max_iter:

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
        m_proposed = subproblem(x_proposed, fun)

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
        k += 1

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            warnflag = 0
            break

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

    if not subproblem.hess_prod:
        result['hess'] = m.hess

    if return_all:
        result['allvecs'] = allvecs

    return result
"""
Dog-leg trust-region optimization.

Code ported from SciPy to PyTorch

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.
"""
import torch
from torch.linalg import norm

from .base import _minimize_trust_region, BaseQuadraticSubproblem


def _minimize_dogleg(
        fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    the dog-leg trust-region algorithm.

    .. warning::
        The Hessian is required to be positive definite at all times;
        otherwise this algorithm will fail.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    References
    ----------
    .. [1] Jorge Nocedal and Stephen Wright,
           Numerical Optimization, second edition,
           Springer-Verlag, 2006, page 73.

    """
    return _minimize_trust_region(fun, x0,
                                  subproblem=DoglegSubproblem,
                                  **trust_region_options)


class DoglegSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by the dogleg method"""
    hess_prod = False

    def cauchy_point(self):
        """
        The Cauchy point is minimal along the direction of steepest descent.
        """
        if self._cauchy_point is None:
            g = self.jac
            Bg = self.hessp(g)
            self._cauchy_point = -(g.dot(g) / g.dot(Bg)) * g
        return self._cauchy_point

    def newton_point(self):
        """
        The Newton point is a global minimum of the approximate function.
        """
        if self._newton_point is None:
            p = -torch.cholesky_solve(self.jac.view(-1,1),
                                      torch.linalg.cholesky(self.hess))
            self._newton_point = p.view(-1)
        return self._newton_point

    def solve(self, trust_radius):
        """Solve quadratic subproblem"""

        # Compute the Newton point.
        # This is the optimum for the quadratic model function.
        # If it is inside the trust radius then return this point.
        p_best = self.newton_point()
        if norm(p_best) < trust_radius:
            hits_boundary = False
            return p_best, hits_boundary

        # Compute the Cauchy point.
        # This is the predicted optimum along the direction of steepest descent.
        p_u = self.cauchy_point()

        # If the Cauchy point is outside the trust region,
        # then return the point where the path intersects the boundary.
        p_u_norm = norm(p_u)
        if p_u_norm >= trust_radius:
            p_boundary = p_u * (trust_radius / p_u_norm)
            hits_boundary = True
            return p_boundary, hits_boundary

        # Compute the intersection of the trust region boundary
        # and the line segment connecting the Cauchy and Newton points.
        # This requires solving a quadratic equation.
        # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
        # Solve this for positive time t using the quadratic formula.
        _, tb = self.get_boundaries_intersections(p_u, p_best - p_u,
                                                  trust_radius)
        p_boundary = p_u + tb * (p_best - p_u)
        hits_boundary = True
        return p_boundary, hits_boundary
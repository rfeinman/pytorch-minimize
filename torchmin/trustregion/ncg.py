"""
Newton-CG trust-region optimization.

Code ported from SciPy to PyTorch

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.
"""
import torch
from torch.linalg import norm

from .base import _minimize_trust_region, BaseQuadraticSubproblem


def _minimize_trust_ncg(
        fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    the Newton conjugate gradient trust-region algorithm.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    Notes
    -----
    This is algorithm (7.2) of Nocedal and Wright 2nd edition.
    Only the function that computes the Hessian-vector product is required.
    The Hessian itself is not required, and the Hessian does
    not need to be positive semidefinite.

    """
    return _minimize_trust_region(fun, x0,
                                  subproblem=CGSteihaugSubproblem,
                                  **trust_region_options)


class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""
    hess_prod = True

    def solve(self, trust_radius):
        """Solve the subproblem using a conjugate gradient method.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : Tensor
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        """

        # get the norm of jacobian and define the origin
        p_origin = torch.zeros_like(self.jac)

        # define a default tolerance
        tolerance = self.jac_mag * self.jac_mag.sqrt().clamp(max=0.5)

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.jac_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.jac
        d = -r

        # Search for the min of the approximation of the objective function.
        while True:

            # do an iteration
            Bd = self.hessp(d)
            dBd = d.dot(Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                p_boundary = torch.where(self(pa).lt(self(pb)), pa, pb)
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = r.dot(r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if norm(z_next) >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = r_next.dot(r_next)
            if r_next_squared.sqrt() < tolerance:
                hits_boundary = False
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next
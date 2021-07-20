import numbers
import numpy as np
import torch
from functools import reduce
from torch.optim import Optimizer
from scipy import optimize
from torch._vmap_internals import _vmap
from torch.autograd.functional import (_construct_standard_basis_for,
                                       _grad_postprocess, _tuple_postprocess,
                                       _as_tuple)


def _build_bounds(bounds, params, numel_total):
    if len(bounds) != len(params):
        raise ValueError('bounds must be an iterable with same length as params')

    lb = np.full(numel_total, -np.inf)
    ub = np.full(numel_total, np.inf)
    keep_feasible = np.zeros(numel_total, dtype=np.bool)

    def process_bound(x, numel):
        if isinstance(x, torch.Tensor):
            assert x.numel() == numel
            return x.view(-1).detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            assert x.size == numel
            return x.flatten()
        elif isinstance(x, (bool, numbers.Number)):
            return x
        else:
            raise ValueError('invalid bound value.')

    offset = 0
    for bound, p in zip(bounds, params):
        numel = p.numel()
        if bound is None:
            offset += numel
            continue
        if not isinstance(bound, (list, tuple)) and len(bound) in [2,3]:
            raise ValueError('elements of "bounds" must each be a '
                             'list/tuple of length 2 or 3')
        if bound[0] is None and bound[1] is None:
            raise ValueError('either lower or upper bound must be defined.')
        if bound[0] is not None:
            lb[offset:offset + numel] = process_bound(bound[0], numel)
        if bound[1] is not None:
            ub[offset:offset + numel] = process_bound(bound[1], numel)
        if len(bound) == 3:
            keep_feasible[offset:offset + numel] = process_bound(bound[2], numel)
        offset += numel

    return optimize.Bounds(lb, ub, keep_feasible)


def _jacobian(inputs, outputs):
    """A modified variant of torch.autograd.functional.jacobian for
    pre-computed outputs

    This is only used for nonlinear parameter constraints (if provided)
    """
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs", "jacobian")

    output_numels = tuple(output.numel() for output in outputs)
    grad_outputs = _construct_standard_basis_for(outputs, output_numels)
    with torch.enable_grad():
        flat_outputs = tuple(output.reshape(-1) for output in outputs)

    def vjp(grad_output):
        vj = list(torch.autograd.grad(flat_outputs, inputs, grad_output, allow_unused=True))
        for el_idx, vj_el in enumerate(vj):
            if vj_el is not None:
                continue
            vj[el_idx] = torch.zeros_like(inputs[el_idx])
        return tuple(vj)

    jacobians_of_flat_output = _vmap(vjp)(grad_outputs)

    jacobian_input_output = []
    for jac, input_i in zip(jacobians_of_flat_output, inputs):
        jacobian_input_i_output = []
        for jac, output_j in zip(jac.split(output_numels, dim=0), outputs):
            jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
            jacobian_input_i_output.append(jacobian_input_i_output_j)
        jacobian_input_output.append(jacobian_input_i_output)

    jacobian_output_input = tuple(zip(*jacobian_input_output))

    jacobian_output_input = _grad_postprocess(jacobian_output_input, create_graph=False)
    return _tuple_postprocess(jacobian_output_input, (is_outputs_tuple, is_inputs_tuple))


class ScipyMinimizer(Optimizer):
    """A PyTorch optimizer for constrained & unconstrained function
    minimization.

    .. note::
        This optimizer is a wrapper for :func:`scipy.optimize.minimize`.
        It uses autograd behind the scenes to build jacobian & hessian
        callables before invoking scipy. Inputs and objectivs should use
        PyTorch tensors like other routines. CUDA is supported; however,
        data will be transferred back-and-forth between GPU/CPU.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Parameters
    ----------
    params : iterable
        An iterable of :class:`torch.Tensor` s. Specifies what Tensors
        should be optimized.
    method : str
        One of the various optimization methods offered in scipy minimize.
        Defaults to 'bfgs'.
    bounds : iterable, optional
        An iterable of :class:`torch.Tensor` s or :class:`float` s with same
        length as `params`. Specifies boundaries for each parameter.
    constraints : dict, optional
        TODO
    tol : float, optional
        TODO
    options : dict, optional
        TODO

    """
    def __init__(self,
                 params,
                 method='bfgs',
                 bounds=None,
                 constraints=(),  # experimental feature! use with caution
                 tol=None,
                 options=None):
        assert isinstance(method, str)
        method = method.lower()
        defaults = dict(
            method=method,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Minimize doesn't support per-parameter options "
                             "(parameter groups)")
        if constraints != () and method != 'trust-constr':
            raise NotImplementedError("Constraints only currently supported for "
                                      "method='trust-constr'.")

        self._params = self.param_groups[0]['params']
        self._param_bounds = self.param_groups[0]['bounds']
        self._numel_cache = None
        self._bounds_cache = None
        self._result = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _bounds(self):
        if self._param_bounds is None:
            return None
        if self._bounds_cache is None:
            self._bounds_cache = _build_bounds(self._param_bounds, self._params,
                                               self._numel())
        return self._bounds_cache

    def _gather_flat_param(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _set_flat_param(self, value):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.copy_(value[offset:offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    def _build_constraints(self, constraints):
        assert isinstance(constraints, dict)
        assert 'fun' in constraints
        assert 'lb' in constraints or 'ub' in constraints

        to_tensor = lambda x: self._params[0].new_tensor(x)
        to_array = lambda x: x.cpu().numpy()
        fun_ = constraints['fun']
        lb = constraints.get('lb', -np.inf)
        ub = constraints.get('ub', np.inf)
        strict = constraints.get('keep_feasible', False)
        lb = to_array(lb) if torch.is_tensor(lb) else lb
        ub = to_array(ub) if torch.is_tensor(ub) else ub
        strict = to_array(strict) if torch.is_tensor(strict) else strict

        def fun(x):
            self._set_flat_param(to_tensor(x))
            return to_array(fun_())

        def jac(x):
            self._set_flat_param(to_tensor(x))
            with torch.enable_grad():
                output = fun_()

            # this is now a tuple of tensors, one per parameter, each with
            # shape (num_outputs, *param_shape).
            J_seq = _jacobian(inputs=tuple(self._params), outputs=output)

            # flatten and stack the tensors along dim 1 to get our full matrix
            J = torch.cat([elt.view(output.numel(), -1) for elt in J_seq], 1)

            return to_array(J)

        return optimize.NonlinearConstraint(fun, lb, ub, jac=jac, keep_feasible=strict)

    @torch.no_grad()
    def step(self, closure):
        """Perform an optimization step.

        Parameters
        ----------
        closure : callable
            A function that re-evaluates the model and returns the loss.
            See the `closure instructions
            <https://pytorch.org/docs/stable/optim.html#optimizer-step-closure>`_
            from PyTorch Optimizer docs for areference on how to construct
            this callable.
        """
        # sanity check
        assert len(self.param_groups) == 1

        # functions to convert numpy -> torch and torch -> numpy
        to_tensor = lambda x: self._params[0].new_tensor(x)
        to_array = lambda x: x.cpu().numpy()

        # optimizer settings
        group = self.param_groups[0]
        method = group['method']
        bounds = self._bounds()
        constraints = group['constraints']
        tol = group['tol']
        options = group['options']

        # build constraints (if provided)
        if constraints != ():
            constraints = self._build_constraints(constraints)

        # build objective
        def fun(x):
            x = to_tensor(x)
            self._set_flat_param(x)
            with torch.enable_grad():
                loss = closure()
            grad = self._gather_flat_grad()
            return float(loss), to_array(grad)

        # initial value (numpy array)
        x0 = to_array(self._gather_flat_param())

        # optimize
        self._result = optimize.minimize(
            fun, x0, method=method, jac=True, bounds=bounds,
            constraints=constraints, tol=tol, options=options
        )

        # set final param
        self._set_flat_param(to_tensor(self._result.x))

        return to_tensor(self._result.fun)

# **** Optimization Utilities ****
#
# This module contains general utilies for optimization such as
# `_status_message` and `OptimizeResult` (coming soon).


# standard status messages of optimizers (derived from SciPy)
_status_message = {
    'success': 'Optimization terminated successfully.',
    'maxfev': 'Maximum number of function evaluations has been exceeded.',
    'maxiter': 'Maximum number of iterations has been exceeded.',
    'pr_loss': 'Desired error not necessarily achieved due to precision loss.',
    'nan': 'NaN result encountered.',
    'out_of_bounds': 'The result is outside of the provided bounds.',
    'callback_stop': 'Stopped by the user through the callback function.',
}

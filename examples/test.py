# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:53:13 2022

@author: X226840
"""

import torch
from torchmin import minimize

def rosen(x):
    return torch.sum(100*(x[..., 1:] - x[..., :-1]**2)**2 
                     + (1 - x[..., :-1])**2)

# initial point
x0 = torch.tensor([1., 8.])

# Select from the following methods:
#  ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
#   'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']

# BFGS
result = minimize(rosen, x0, method='bfgs')

# Newton Conjugate Gradient
result = minimize(rosen, x0, method='newton-cg')

# Newton Exact
result = minimize(rosen, x0, method='newton-exact')

# Levenberg-Marquardt Exact
result = minimize(rosen, x0, method = 'lm-exact')
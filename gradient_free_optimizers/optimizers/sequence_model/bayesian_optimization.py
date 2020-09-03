# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist

from .exp_imp_based_opt import ExpectedImprovementBasedOptimization

from .surrogate_models import (
    GPR_linear,
    GPR,
)

gaussian_process = {"gp_nonlinear": GPR(), "gp_linear": GPR_linear()}


class BayesianOptimizer(ExpectedImprovementBasedOptimization):
    def __init__(self, search_space, gpr=gaussian_process["gp_nonlinear"], **kwargs):
        super().__init__(search_space, **kwargs)
        self.regr = gpr


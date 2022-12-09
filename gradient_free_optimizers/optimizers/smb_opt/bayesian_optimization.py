# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from scipy.stats import norm


from .smbo import SMBO
from .surrogate_models import (
    GPR_linear,
    GPR,
)
from .acquisition_function import ExpectedImprovement


gaussian_process = {"gp_nonlinear": GPR(), "gp_linear": GPR_linear()}


def normalize(array):
    num = array - array.min()
    den = array.max() - array.min()

    if den == 0:
        return np.random.random_sample(array.shape)
    else:
        return ((num / den) + 0) / 1


class BayesianOptimizer(SMBO):
    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"
    __name__ = "BayesianOptimizer"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        *args,
        gpr=gaussian_process["gp_nonlinear"],
        xi=0.03,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gpr = gpr
        self.regr = gpr
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.warnings = warnings

        self.init_warm_start_smbo()

    def finish_initialization(self):
        self.all_pos_comb = self._all_possible_pos()
        return super().finish_initialization()

    def _expected_improvement(self):
        self.pos_comb = self._sampling(self.all_pos_comb)

        acqu_func = ExpectedImprovement(self.regr, self.pos_comb, self.xi)
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _training(self):
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

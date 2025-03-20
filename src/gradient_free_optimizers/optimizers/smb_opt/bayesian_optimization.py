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
    array_min = array.min()
    array_max = array.max()
    range_ = array_max - array_min

    if range_ == 0:
        return np.random.random_sample(array.shape)
    else:
        return (array - array_min) / range_


class BayesianOptimizer(SMBO):
    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"
    __name__ = "BayesianOptimizer"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        replacement=True,
        gpr=gaussian_process["gp_nonlinear"],
        xi=0.03,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

        self.gpr = gpr
        self.regr = gpr
        self.xi = xi

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

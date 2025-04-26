# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from scipy.stats import norm

from ..smb_opt.smbo import SMBO
from ..smb_opt.surrogate_models import EnsembleRegressor
from ..smb_opt.acquisition_function import ExpectedImprovement


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


def normalize(array):
    num = array - array.min()
    den = array.max() - array.min()

    if den == 0:
        return np.random.random_sample(array.shape)
    else:
        return ((num / den) + 0) / 1


class EnsembleOptimizer(SMBO):
    name = "Ensemble Optimizer"

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        epsilon=0.03,
        distribution="normal",
        n_neighbours=3,
        estimators=[
            GradientBoostingRegressor(n_estimators=5),
            # DecisionTreeRegressor(),
            # MLPRegressor(),
            GaussianProcessRegressor(),
        ],
        xi=0.01,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        replacement=True,
        warnings=100000000,
        **kwargs,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,  #
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )
        self.estimators = estimators
        self.regr = EnsembleRegressor(estimators)
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

        if len(Y_sample) == 0:
            return self.move_random()

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

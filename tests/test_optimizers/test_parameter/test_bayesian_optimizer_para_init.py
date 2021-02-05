# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import BayesianOptimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from ._base_para_test import _base_para_test_func
from gradient_free_optimizers import RandomSearchOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-10, 11, 1)}

search_space2 = {"x1": np.arange(-10, 51, 1)}
search_space3 = {"x1": np.arange(-50, 11, 1)}


warm_start_smbo = (
    np.array([[-10, -10], [30, 30], [0, 0]]),
    np.array([-1, 0, 1]),
)


opt1 = RandomSearchOptimizer(search_space)
opt2 = RandomSearchOptimizer(search_space2)
opt3 = RandomSearchOptimizer(search_space3)

opt1.search(objective_function, n_iter=30)
opt2.search(objective_function, n_iter=30)
opt3.search(objective_function, n_iter=30)

search_data1 = opt1.results
search_data2 = opt2.results
search_data3 = opt3.results


class GPR:
    def __init__(self):
        nu_param = 0.5
        matern = Matern(
            # length_scale=length_scale_param,
            # length_scale_bounds=length_scale_bounds_param,
            nu=nu_param,
        )

        self.gpr = GaussianProcessRegressor(
            kernel=matern + RBF() + WhiteKernel(), n_restarts_optimizer=0
        )

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)


bayesian_optimizer_para = [
    ({"gpr": GPR()}),
    ({"xi": 0.001}),
    ({"xi": 0.5}),
    ({"xi": 0.9}),
    ({"warm_start_smbo": None}),
    ({"warm_start_smbo": search_data1}),
    ({"warm_start_smbo": search_data2}),
    ({"warm_start_smbo": search_data3}),
    ({"rand_rest_p": 0}),
    ({"rand_rest_p": 0.5}),
    ({"rand_rest_p": 1}),
    ({"rand_rest_p": 10}),
]


pytest_wrapper = ("opt_para", bayesian_optimizer_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_hill_climbing_para(opt_para):
    _base_para_test_func(opt_para, BayesianOptimizer)

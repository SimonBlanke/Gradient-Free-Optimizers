# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .exp_imp_based_opt import ExpectedImprovementBasedOptimization
from .surrogate_models import EnsembleRegressor


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


class EnsembleOptimizer(ExpectedImprovementBasedOptimization):
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        estimators=[
            GradientBoostingRegressor(n_estimators=5),
            # DecisionTreeRegressor(),
            # MLPRegressor(),
            GaussianProcessRegressor(),
        ],
        xi=0.01,
        warm_start_smbo=None,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space, initialize)
        self.regr = EnsembleRegressor(estimators)
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.rand_rest_p = rand_rest_p


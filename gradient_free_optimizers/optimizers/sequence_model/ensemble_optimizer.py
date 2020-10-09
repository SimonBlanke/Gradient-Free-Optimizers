# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .exp_imp_based_opt import ExpectedImprovementBasedOptimization
from .surrogate_models import EnsembleRegressor


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor


class EnsembleOptimizer(ExpectedImprovementBasedOptimization):
    def __init__(
        self,
        search_space,
        estimators=[
            GradientBoostingRegressor(n_estimators=10),
            SVR(),
            DecisionTreeRegressor(),
            GaussianProcessRegressor(),
        ],
        **kwargs,
    ):
        super().__init__(search_space)
        self.regr = EnsembleRegressor(estimators)

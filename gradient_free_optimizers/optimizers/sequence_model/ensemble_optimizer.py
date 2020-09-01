# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .bayesian_optimization import BayesianOptimizer
from .surrogate_models import EnsembleRegressor


from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor


class EnsembleOptimizer(BayesianOptimizer):
    def __init__(
        self,
        search_space,
        estimators=[
            GradientBoostingRegressor(n_estimators=10),
            SVR(),
            DecisionTreeRegressor(),
            GaussianProcessRegressor(),
        ],
    ):
        super().__init__(search_space)
        self.regr = EnsembleRegressor(estimators)

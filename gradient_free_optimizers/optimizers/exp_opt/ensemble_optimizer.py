# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..smb_opt.exp_imp_based_opt import ExpectedImprovementBasedOptimization
from ..smb_opt.surrogate_models import EnsembleRegressor


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


class EnsembleOptimizer(ExpectedImprovementBasedOptimization):
    name = "Ensemble Optimizer"

    def __init__(
        self,
        *args,
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
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.estimators = estimators
        self.regr = EnsembleRegressor(estimators)
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.warnings = warnings

        self.init_warm_start_smbo()

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .exp_imp_based_opt import ExpectedImprovementBasedOptimization
from .surrogate_models import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

tree_regressor_dict = {
    "random_forest": RandomForestRegressor,
    "extra_tree": ExtraTreesRegressor,
    "gradient_boost": GradientBoostingRegressor,
}


class ForestOptimizer(ExpectedImprovementBasedOptimization):
    name = "Forest Optimization"
    """Based on the forest-optimizer in the scikit-optimize package"""

    def __init__(
        self,
        *args,
        tree_regressor="extra_tree",
        tree_para={"n_estimators": 100},
        xi=0.03,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tree_regressor = tree_regressor
        self.tree_para = tree_para
        self.regr = tree_regressor_dict[tree_regressor](**self.tree_para)
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.warnings = warnings

        self.init_warm_start_smbo()

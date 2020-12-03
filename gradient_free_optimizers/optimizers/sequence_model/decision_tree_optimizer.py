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
    "random_forest": RandomForestRegressor(n_estimators=5),
    "extra_tree": ExtraTreesRegressor(n_estimators=5),
    "gradient_boost": GradientBoostingRegressor(n_estimators=5),
}


class DecisionTreeOptimizer(ExpectedImprovementBasedOptimization):
    """Based on the forest-optimizer in the scikit-optimize package"""

    def __init__(
        self,
        search_space,
        tree_regressor="extra_tree",
        xi=0.01,
        warm_start_smbo=None,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space)
        self.tree_regressor = tree_regressor
        self.regr = tree_regressor_dict[tree_regressor]
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.rand_rest_p = rand_rest_p

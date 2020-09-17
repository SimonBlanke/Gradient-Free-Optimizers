# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .exp_imp_based_opt import ExpectedImprovementBasedOptimization
from .surrogate_models import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

tree_regressor_dict = {
    "random_forest": RandomForestRegressor(n_estimators=10),
    "extra_tree": ExtraTreesRegressor(n_estimators=10),
}


class DecisionTreeOptimizer(ExpectedImprovementBasedOptimization):
    """Based on the forest-optimizer in the scikit-optimize package"""

    def __init__(self, search_space, tree_regressor="random_forest", **kwargs):
        super().__init__(search_space)
        self.regr = tree_regressor_dict[tree_regressor]

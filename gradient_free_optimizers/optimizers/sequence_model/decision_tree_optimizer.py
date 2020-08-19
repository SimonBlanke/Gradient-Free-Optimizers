# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .bayesian_optimization import BayesianOptimizer
from .surrogate_models import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

tree_regressor_dict = {
    "random_forest": RandomForestRegressor(),
    "extra_tree": ExtraTreesRegressor(),
}


class DecisionTreeOptimizer(BayesianOptimizer):
    """Based on the forest-optimizer in the scikit-optimize package"""

    def __init__(self, search_space, tree_regressor="random_forest", **kwargs):
        super().__init__(search_space, **kwargs)
        self.regr = tree_regressor_dict[tree_regressor]

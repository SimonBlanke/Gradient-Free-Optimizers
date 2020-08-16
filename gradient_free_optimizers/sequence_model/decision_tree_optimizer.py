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

    def __init__(self, space_dim, tree_regressor="random_forest"):
        super().__init__(space_dim)
        self.regr = tree_regressor_dict[tree_regressor]

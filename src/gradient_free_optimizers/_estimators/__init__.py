# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Internal native estimators for surrogate modeling (no sklearn dependency).

These are not part of the public API and should only be used internally.
"""

from ._decision_tree_regressor import DecisionTreeRegressor as DecisionTreeRegressor
from ._extra_trees_regressor import ExtraTreesRegressor as ExtraTreesRegressor
from ._gaussian_process_regressor import (
    GaussianProcessRegressor as GaussianProcessRegressor,
)
from ._gradient_boosting_regressor import (
    GradientBoostingRegressor as GradientBoostingRegressor,
)
from ._kernel_density_estimator import KernelDensityEstimator as KernelDensityEstimator
from ._random_forest_regressor import RandomForestRegressor as RandomForestRegressor

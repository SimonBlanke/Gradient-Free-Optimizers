# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Internal native estimators for surrogate modeling (no sklearn dependency).

These are not part of the public API and should only be used internally.
"""

from ._gaussian_process_regressor import GaussianProcessRegressor
from ._kernel_density_estimator import KernelDensityEstimator

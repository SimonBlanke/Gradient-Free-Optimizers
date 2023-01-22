import numpy as np
from gradient_free_optimizers import TreeStructuredParzenEstimators


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = TreeStructuredParzenEstimators(
    search_space,
    gamma_tpe=0.5,
)
opt.search(sphere_function, n_iter=50)

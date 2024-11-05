import numpy as np
from gradient_free_optimizers import DownhillSimplexOptimizer


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = DownhillSimplexOptimizer(
    search_space,
    alpha=1.2,
    gamma=1.1,
    beta=0.8,
    sigma=1,
)
opt.search(sphere_function, n_iter=10000)

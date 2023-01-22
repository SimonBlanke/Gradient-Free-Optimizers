import numpy as np
from gradient_free_optimizers import HillClimbingOptimizer


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = HillClimbingOptimizer(
    search_space,
    epsilon=0.1,
    n_neighbours=5,
    distribution="laplace",
)
opt.search(sphere_function, n_iter=100000)

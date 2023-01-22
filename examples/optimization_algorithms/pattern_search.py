import numpy as np
from gradient_free_optimizers import PatternSearch


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = PatternSearch(
    search_space,
    n_positions=2,
    pattern_size=0.5,
    reduction=0.99,
)
opt.search(sphere_function, n_iter=10000)

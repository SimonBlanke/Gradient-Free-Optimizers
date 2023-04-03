import numpy as np
from gradient_free_optimizers import SpiralOptimization


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = SpiralOptimization(
    search_space,
    population=25,
    decay_rate=1.05,
)
opt.search(sphere_function, n_iter=10000)

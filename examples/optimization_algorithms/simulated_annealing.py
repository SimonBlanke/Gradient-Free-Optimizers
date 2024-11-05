import numpy as np
from gradient_free_optimizers import SimulatedAnnealingOptimizer


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = SimulatedAnnealingOptimizer(
    search_space,
    start_temp=1.2,
    annealing_rate=0.99,
)
opt.search(sphere_function, n_iter=10000)

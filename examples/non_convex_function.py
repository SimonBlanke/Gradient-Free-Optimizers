import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def ackley_function(pos_new):
    x = pos_new["x1"]
    y = pos_new["x2"]

    a1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
    a2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    score = a1 + a2 + 20
    return -score


search_space = {
    "x1": np.arange(-100, 101, 0.1),
    "x2": np.arange(-100, 101, 0.1),
}

opt = RandomSearchOptimizer(search_space)
opt.search(ackley_function, n_iter=30000)

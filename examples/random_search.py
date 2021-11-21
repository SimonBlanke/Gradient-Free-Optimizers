import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def convex_function(pos_new):
    score = -(pos_new["x1"] * pos_new["x1"] + pos_new["x2"] * pos_new["x2"])
    return score


search_space = {
    "x1": np.arange(-100, 101, 0.1),
    "x2": np.arange(-100, 101, 0.1),
}

opt = RandomSearchOptimizer(search_space)

opt.search(convex_function, n_iter=300000)
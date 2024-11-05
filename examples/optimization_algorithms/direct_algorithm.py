import numpy as np
from gradient_free_optimizers import DirectAlgorithm


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = DirectAlgorithm(search_space)
opt.search(sphere_function, n_iter=300)

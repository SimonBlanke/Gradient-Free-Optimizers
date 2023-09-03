import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def ackley_function(para):
    x, y = para["x"], para["y"]

    loss = (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.exp(1)
        + 20
    )

    return -loss


search_space = {
    "x": np.arange(-10, 10, 0.01),
    "y": np.arange(-10, 10, 0.01),
}


opt = RandomSearchOptimizer(search_space)
opt.search(ackley_function, n_iter=30)

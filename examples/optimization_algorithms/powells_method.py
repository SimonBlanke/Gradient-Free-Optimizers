import numpy as np
from gradient_free_optimizers import PowellsConjugateDirectionMethod


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = PowellsConjugateDirectionMethod(
    search_space,
    iters_p_dim=20,
)
opt.search(sphere_function, n_iter=10000)

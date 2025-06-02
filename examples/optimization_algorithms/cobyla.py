import numpy as np
from gradient_free_optimizers import COBYLA


def sphere_function(para: np.array):
    x = para[0]
    y = para[1]

    return -(x * x + y * y)

def constraint_1(para):
    return para[0] > -5

search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = COBYLA(
    search_space=search_space,
    rho_beg=1.0,
    rho_end=0.01,
    x_0=np.array([0.0, 0.0]),
    constraints=[constraint_1]
)
opt.search(sphere_function, n_iter=10)

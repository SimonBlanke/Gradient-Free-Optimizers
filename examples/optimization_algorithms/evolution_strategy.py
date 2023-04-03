import numpy as np
from gradient_free_optimizers import EvolutionStrategyOptimizer


def sphere_function(para):
    x = para["x"]
    y = para["y"]

    return -(x * x + y * y)


search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

opt = EvolutionStrategyOptimizer(
    search_space,
    population=25,
    mutation_rate=0.5,
    crossover_rate=0.5,
)
opt.search(sphere_function, n_iter=10000)

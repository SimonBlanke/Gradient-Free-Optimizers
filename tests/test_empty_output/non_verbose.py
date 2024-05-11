def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import numpy as np
from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    PowellsMethod,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    PatternSearch,
    DirectAlgorithm,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    EvolutionStrategyOptimizer,
    LipschitzOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)

optimizers = [
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    PowellsMethod,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    PatternSearch,
    DirectAlgorithm,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    EvolutionStrategyOptimizer,
    LipschitzOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
]


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
    "x": np.arange(-10, 10, 1),
    "y": np.arange(-10, 10, 1),
}

for optimizer in optimizers:
    opt = optimizer(search_space)
    opt.search(ackley_function, n_iter=15, verbosity=False)

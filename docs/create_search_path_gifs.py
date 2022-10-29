def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import os
import numpy as np


from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    PowellsMethod,
    PatternSearch,
    LipschitzOptimizer,
    DirectAlgorithm,
    RandomAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)
from surfaces.test_functions import SphereFunction, AckleyFunction

from search_path_gif import search_path_gif

here = os.path.dirname(os.path.abspath(__file__))

search_space = {
    "x0": np.arange(-2, 8, 0.1),
    "x1": np.arange(-2, 8, 0.1),
}

initialize_1 = {
    "warm_start": [{"x0": 7, "x1": 7}],
}
initialize_pop = {"vertices": 4}
random_state = 23

sphere_function = SphereFunction(n_dim=2)
ackley_function = AckleyFunction()

objective_function_l = [sphere_function, ackley_function]

optimizer_l = [
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    PowellsMethod,
    PatternSearch,
    LipschitzOptimizer,
    DirectAlgorithm,
    RandomAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
]
"""

"""

for objective_function in objective_function_l:
    for optimizer in optimizer_l:
        if optimizer.computationally_expensive:
            n_iter = 50
        else:
            n_iter = 150

        if optimizer.optimizer_type == "population":
            initialize = initialize_pop
        else:
            initialize = initialize_1

        para_dict = {
            "path": os.path.join(here, "gifs"),
            "optimizer": optimizer,
            "opt_para": {},
            "name": optimizer._name_ + "_" + objective_function._name_ + "_" + ".gif",
            "n_iter": n_iter,
            "objective_function": objective_function,
            "search_space": search_space,
            "initialize": initialize,
            "random_state": random_state,
        }
        search_path_gif(**para_dict)

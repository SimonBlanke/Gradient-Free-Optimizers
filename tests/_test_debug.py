import numpy as np
from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
    EnsembleOptimizer,
)

# check if there are any debug-prints left in code


optimizer_list = [
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
    EnsembleOptimizer,
]


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 5, 1),
}


for optimizer in optimizer_list:
    opt0 = optimizer(search_space)
    opt0.search(objective_function, n_iter=15, verbosity=False, memory=False)

    opt1 = optimizer(search_space)
    opt1.search(objective_function, n_iter=15, verbosity=False)

    opt2 = optimizer(search_space)
    opt2.search(
        objective_function,
        n_iter=15,
        verbosity=False,
        memory_warm_start=opt1.search_data,
    )

    opt3 = optimizer(search_space, initialize={"warm_start": [{"x1": 1}]})
    opt3.search(
        objective_function,
        n_iter=15,
        verbosity=False,
    )

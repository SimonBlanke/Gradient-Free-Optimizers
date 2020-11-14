from tqdm import tqdm
import numpy as np
import pandas as pd

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    TabuOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    DecisionTreeOptimizer,
)


optimizer_dict = {
    "HillClimbing": HillClimbingOptimizer,
    "StochasticHillClimbingOptimizer": StochasticHillClimbingOptimizer,
}


def create_convergence_data(optimizer_key):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-100, 101, 1)}
    initialize = {"vertices": 2}

    n_opts = 30
    n_iter = 100

    scores_list = []
    for rnd_st in tqdm(range(n_opts)):
        opt = optimizer_dict[optimizer_key](search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            verbosity=False,
            initialize=initialize,
        )

        scores_list.append(opt.results["score"])

    convergence_data = pd.concat(scores_list, axis=1)
    convergence_data.to_csv(
        "./data/" + optimizer_key + "_convergence_data", index=False
    )


for opt_key in optimizer_dict.keys():
    create_convergence_data(opt_key)

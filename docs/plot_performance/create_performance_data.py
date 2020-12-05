import time
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
    EnsembleOptimizer,
)


n_inits = 4

optimizer_dict = {
    "Hill climbing": HillClimbingOptimizer,
    "Stochastic hill climbing": StochasticHillClimbingOptimizer,
    "Tabu search": TabuOptimizer,
    "Random search": RandomSearchOptimizer,
    "Random restart hill climbing": RandomRestartHillClimbingOptimizer,
    "Random annealing": RandomAnnealingOptimizer,
    "Simulated annealing": SimulatedAnnealingOptimizer,
    "Parallel tempering": ParallelTemperingOptimizer,
    "Particle swarm optimizer": ParticleSwarmOptimizer,
    "Evolution strategy": EvolutionStrategyOptimizer,
    # "Bayesian optimizer": BayesianOptimizer,
    # "Tree structured parzen estimators": TreeStructuredParzenEstimators,
    # "Decision tree optimizer": DecisionTreeOptimizer,
    # "Ensemble optimizer": EnsembleOptimizer,
}


def objective_function(pos_new):
    score = -(pos_new["x1"] * pos_new["x1"] + pos_new["x2"] * pos_new["x2"])
    return score


search_space = {"x1": np.arange(-10, 11, 0.1), "x2": np.arange(-10, 11, 0.1)}

runs = 30


def create_performance_data(
    study_name, objective_function, search_space, n_iter
):
    results = []

    for opt_name in tqdm(optimizer_dict.keys()):
        total_time_list = []
        eval_time_list = []
        iter_time_list = []

        for random_state in tqdm(range(runs)):

            c_time = time.time()
            opt = optimizer_dict[opt_name](search_space)
            opt.search(
                objective_function,
                n_iter=n_iter,
                verbosity=False,
                random_state=random_state,
            )

            total_time = time.time() - c_time
            eval_time = np.array(opt.eval_times).sum()
            iter_time = np.array(opt.iter_times).sum()

            total_time_list.append(total_time)
            eval_time_list.append(eval_time)
            iter_time_list.append(iter_time)

        total_time_mean = np.array(total_time_list).mean()
        eval_time_mean = np.array(eval_time_list).mean()
        iter_time_mean = np.array(iter_time_list).mean()

        total_time_std = np.array(total_time_list).std()
        eval_time_std = np.array(eval_time_list).std()
        iter_time_std = np.array(iter_time_list).std()

        results.append(
            [
                total_time_mean,
                total_time_std,
                eval_time_mean,
                eval_time_std,
                iter_time_mean,
                iter_time_std,
            ]
        )

    index = [
        "total_time_mean",
        "total_time_std",
        "eval_time_mean",
        "eval_time_std",
        "iter_time_mean",
        "iter_time_std",
    ]
    columns = list(optimizer_dict.keys())

    results = np.array(results).T
    results = pd.DataFrame(results, columns=columns, index=index)
    results.to_csv("./_data/" + study_name + ".csv")


create_performance_data(
    "simple function", objective_function, search_space, n_iter=50
)


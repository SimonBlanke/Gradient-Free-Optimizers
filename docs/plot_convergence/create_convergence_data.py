from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
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

n_inits = 2


optimizer_dict = {
    "HillClimbing": HillClimbingOptimizer,
    "StochasticHillClimbing": StochasticHillClimbingOptimizer,
}

"""


x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-(x ** 2) / 9.0)
f2 = interp1d(x, y, kind="cubic")

print("f2", f2)
"""


def create_convergence_data(optimizer_key):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-100, 101, 1)}
    initialize = {"vertices": 2}

    n_opts = 33
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

        best_scores = opt.p_bar.convergence_data
        scores_list.append(best_scores)

    scores_np = np.array(scores_list).T

    scores_mean = scores_np.mean(axis=1)
    scores_std = scores_np.std(axis=1)

    results = np.array([scores_mean, scores_std]).T

    convergence_data = pd.DataFrame(
        results, columns=["scores_mean", "scores_std"],
    )

    convergence_data.to_csv(
        "./_data/" + optimizer_key + "_convergence_data.csv", index=False
    )


for opt_key in optimizer_dict.keys():
    create_convergence_data(opt_key)


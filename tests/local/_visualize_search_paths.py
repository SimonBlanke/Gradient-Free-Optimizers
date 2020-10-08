import os
import numpy as np
import matplotlib.pyplot as plt
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
    "HillClimbing": (HillClimbingOptimizer, 1),
    "StochasticHillClimbing": (StochasticHillClimbingOptimizer, 1),
    "TabuSearch": (TabuOptimizer, 1),
    "RandomSearch": (RandomSearchOptimizer, 1),
    "RandomRestartHillClimbing": (RandomRestartHillClimbingOptimizer, 1),
    "RandomAnnealing": (RandomAnnealingOptimizer, 1),
    "SimulatedAnnealing": (SimulatedAnnealingOptimizer, 1),
    "ParallelTempering": (ParallelTemperingOptimizer, n_inits),
    "ParticleSwarm": (ParticleSwarmOptimizer, n_inits),
    "EvolutionStrategy": (EvolutionStrategyOptimizer, n_inits),
    "Bayesian": (BayesianOptimizer, 1),
    "TPE": (TreeStructuredParzenEstimators, 1),
    "DecisionTree": (DecisionTreeOptimizer, 1),
    "Ensemble": (EnsembleOptimizer, 1),
}


def objective_function(pos_new):
    score = -(pos_new["x1"] * pos_new["x1"] + pos_new["x2"] * pos_new["x2"])
    return score


search_space = {"x1": np.arange(-100, 100, 1), "x2": np.arange(-100, 100, 1)}


def plot_search_path(optimizer_key):
    opt_class, n_inits = optimizer_dict[optimizer_key]
    opt = opt_class(search_space)

    opt.search(
        objective_function,
        n_iter=50,
        random_state=0,
        memory=False,
        verbosity={"progress_bar": True, "print_results": False},
        initialize={"vertices": n_inits},
    )

    optimizers = opt.optimizers

    print(optimizers, "\n")

    plt.figure(figsize=(5.5, 4.7))
    plt.set_cmap("jet")

    for n, opt_ in enumerate(optimizers):
        pos_list = np.array(opt_.pos_new_list)
        score_list = np.array(opt_.score_new_list)

        # print("\npos_list\n", pos_list, "\n", len(pos_list))
        # print("score_list\n", score_list, "\n", len(score_list))

        plt.plot(
            pos_list[:, 0],
            pos_list[:, 1],
            linestyle="--",
            marker=",",
            color="black",
            alpha=0.33,
            label=n,
        )
        plt.scatter(
            pos_list[:, 0],
            pos_list[:, 1],
            c=score_list,
            marker="H",
            s=5,
            vmin=-1000,
            vmax=0,
            label=n,
        )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xlim((0, 200))
    plt.ylim((0, 200))
    plt.colorbar()
    # plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.abspath(os.path.dirname(__file__))
        + "/plots/temp/"
        + optimizer_key
        + "_path.png",
        dpi=400,
    )


for key in optimizer_dict.keys():
    print(key)
    plot_search_path(key)

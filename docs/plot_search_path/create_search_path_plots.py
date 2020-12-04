import numpy as np
import pandas as pd
from tqdm import tqdm
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
from gradient_free_optimizers.converter import Converter

one_init = 1
two_init = 2
six_init = 6
n_inits = 4

"""
    "Stochastic hill climbing": (StochasticHillClimbingOptimizer, one_init, 0),
    "Tabu search": (TabuOptimizer, one_init, 0),
    "Random search": (RandomSearchOptimizer, one_init, 1),
    "Random restart hill climbing": (
        RandomRestartHillClimbingOptimizer,
        one_init,
        9,
    ),
    "Random annealing": (RandomAnnealingOptimizer, one_init, 1),
    "Simulated annealing": (SimulatedAnnealingOptimizer, one_init, 0),
    "Parallel tempering": (ParallelTemperingOptimizer, two_init, 0),
    "Particle swarm optimizer": (ParticleSwarmOptimizer, n_inits, 0),
    "Evolution strategy": (EvolutionStrategyOptimizer, n_inits, 0),
    "Bayesian optimizer": (BayesianOptimizer, six_init, 0),
    "Tree structured parzen estimators": (
        TreeStructuredParzenEstimators,
        six_init,
        0,
    ),
    "Decision tree optimizer": (DecisionTreeOptimizer, six_init, 0),
    "Ensemble optimizer": (EnsembleOptimizer, six_init, 0),
"""

optimizer_dict = {
    "Hill climbing": (HillClimbingOptimizer, one_init, 0),
}


def plot_search_path(
    optimizer_key,
    n_iter,
    objective_function,
    objective_function_np,
    search_space,
):
    opt_class, n_inits, random_state = optimizer_dict[optimizer_key]
    opt = opt_class(search_space, rand_rest_p=0)

    opt.search(
        objective_function,
        n_iter=n_iter,
        random_state=random_state,
        memory=False,
        verbosity=False,
        initialize={"vertices": n_inits},
    )

    conv = Converter(search_space)

    plt.figure(figsize=(10, 8))
    plt.set_cmap("jet_r")

    x_all, y_all = search_space["x"], search_space["y"]
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_function_np(xi, yi)

    plt.imshow(
        zi,
        alpha=0.15,
        # vmin=z.min(),
        # vmax=z.max(),
        # origin="lower",
        extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
    )

    # print("\n Results \n", opt.results)

    for n, opt_ in enumerate(tqdm(opt.optimizers)):
        pos_list = np.array(opt_.pos_new_list)
        score_list = np.array(opt_.score_new_list)

        values_list = conv.positions2values(pos_list)
        values_list = np.array(values_list)

        plt.plot(
            values_list[:, 0],
            values_list[:, 1],
            linestyle="--",
            marker=",",
            color="black",
            alpha=0.33,
            label=n,
            linewidth=0.5,
        )
        plt.scatter(
            values_list[:, 0],
            values_list[:, 1],
            c=score_list,
            marker="H",
            s=15,
            vmin=-20000,
            vmax=0,
            label=n,
            edgecolors="black",
            linewidth=0.3,
        )

    plt.xlabel("x")
    plt.ylabel("y")

    nth_iteration = "\n\nnth Iteration: " + str(n_iter)

    plt.title(optimizer_key + nth_iteration)

    plt.xlim((-101, 101))
    plt.ylim((-101, 101))
    plt.colorbar()
    # plt.legend(loc="upper left", bbox_to_anchor=(-0.10, 1.2))

    plt.tight_layout()
    plt.savefig(
        "./plots/"
        + str(opt.__class__.__name__)
        + "_"
        + "{0:0=2d}".format(n_iter)
        + ".jpg",
        dpi=300,
    )
    plt.close()


# n_iter = 50


def objective_function(pos_new):
    score = -(pos_new["x"] * pos_new["x"] + pos_new["y"] * pos_new["y"])
    return score


def objective_function_np(x1, x2):
    score = -(x1 * x1 + x2 * x2)
    return score


search_space = {"x": np.arange(-100, 101, 1), "y": np.arange(-100, 101, 1)}

n_iter_list = range(1, 51)

for optimizer_key in optimizer_dict.keys():
    print(optimizer_key)

    for n_iter in tqdm(n_iter_list):
        plot_search_path(
            optimizer_key,
            n_iter,
            objective_function,
            objective_function_np,
            search_space,
        )

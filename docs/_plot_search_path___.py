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
    StochasticTunnelingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    DecisionTreeOptimizer,
)

optimizer_dict = {
    "HillClimbing": HillClimbingOptimizer,
    "StochasticHillClimbing": StochasticHillClimbingOptimizer,
    "TabuSearch": TabuOptimizer,
    "RandomSearch": RandomSearchOptimizer,
    "RandomRestartHillClimbing": RandomRestartHillClimbingOptimizer,
    "RandomAnnealing": RandomAnnealingOptimizer,
    "SimulatedAnnealing": SimulatedAnnealingOptimizer,
    "StochasticTunneling": StochasticTunnelingOptimizer,
    "ParallelTempering": ParallelTemperingOptimizer,
    "ParticleSwarm": ParticleSwarmOptimizer,
    "EvolutionStrategy": EvolutionStrategyOptimizer,
    "Bayesian": BayesianOptimizer,
    "TPE": TreeStructuredParzenEstimators,
    "DecisionTree": DecisionTreeOptimizer,
}


def get_score(pos_new):
    x1 = pos_new[0] - 50
    x2 = pos_new[1] - 50

    a = 20
    b = 0.2
    c = 2 * np.pi

    x1 = x1 / 10
    x2 = x2 / 10

    sum1 = x1 ** 2 + x2 ** 2
    sum2 = np.cos(c * x1) + np.cos(c * x2)

    term1 = -a * np.exp(-b * ((1 / 2.0) * sum1 ** (0.5)))
    term2 = -np.exp((1 / 2.0) * sum2)

    return 1 - (term1 + term2 + a + np.exp(1)) / 10


def plot_search_path(optimizer_key):
    n_iter = 30
    space_dim = np.array([100, 100])
    init_positions = [np.array([10, 10]), np.array([90, 50])]

    pos_list = []
    score_list = []

    opt_class = optimizer_dict[optimizer_key]

    opt = opt_class(init_positions, space_dim, opt_para={})

    for nth_init in range(len(init_positions)):
        pos_new = opt.init_pos(nth_init)
        score_new = get_score(pos_new)
        opt.evaluate(score_new)

        pos_list.append(pos_new)
        score_list.append(score_new)

    for nth_iter in range(len(init_positions), n_iter):
        pos_new = opt.iterate(nth_iter)
        score_new = get_score(pos_new)
        opt.evaluate(score_new)

        pos_list.append(pos_new)
        score_list.append(score_new)

    plt.figure(figsize=(5.5, 4.7))
    plt.set_cmap("jet")

    for positioner in opt.p_list:
        pos_list = np.array(positioner.pos_list)
        score_list = np.array(positioner.score_list)

        plt.plot(
            pos_list[:, 0],
            pos_list[:, 1],
            linestyle="-",
            marker=",",
            color="gray",
            alpha=0.15,
        )

        plt.scatter(
            pos_list[:, 0],
            pos_list[:, 1],
            c=score_list,
            marker="H",
            s=5,
            vmin=0,
            vmax=1,
        )

    for positioner in opt.p_list:
        pos_list = np.array(positioner.pos_current_list)
        score_list = np.array(positioner.score_current_list)

        plt.plot(
            pos_list[:, 0],
            pos_list[:, 1],
            linestyle="--",
            marker=",",
            color="black",
            alpha=0.33,
        )
        plt.scatter(
            pos_list[:, 0],
            pos_list[:, 1],
            c=score_list,
            marker="H",
            s=5,
            vmin=0,
            vmax=1,
        )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("./plots/" + optimizer_key + "_path.png", dpi=200)


for key in optimizer_dict.keys():
    print(key)
    plot_search_path(key)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from hyperactive import Hyperactive

X, y = np.array([0]), np.array([0])
memory = False
n_iter = 100


def sphere_function(para, X_train, y_train):
    loss = []
    for key in para.keys():
        if key == "iteration":
            continue
        loss.append(para[key] * para[key])

    return -np.array(loss).sum()


search_config = {
    sphere_function: {
        "x1": list(np.arange(-3, 3, 0.1)),
        "x2": list(np.arange(-3, 3, 0.1)),
    }
}


def test_HillClimbingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="HillClimbing")

    for epsilon in [0.01, 0.1, 1]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"HillClimbing": {"epsilon": epsilon}},
        )


def test_StochasticHillClimbingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="StochasticHillClimbing")

    for p_down in [0.01, 0.1, 1]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"StochasticHillClimbing": {"p_down": p_down}},
        )


def test_TabuOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="TabuSearch")

    for tabu_memory in [1, 3, 5]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"TabuSearch": {"tabu_memory": tabu_memory}},
        )


def test_RandomSearchOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomSearch")


def test_RandomRestartHillClimbingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomRestartHillClimbing")

    for n_restarts in [3, 5, 20]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"RandomRestartHillClimbing": {"n_restarts": n_restarts}},
        )


def test_RandomAnnealingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomAnnealing")

    for start_temp in [0.1, 1, 10]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"RandomAnnealing": {"start_temp": start_temp}},
        )


def test_SimulatedAnnealingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="SimulatedAnnealing")

    for start_temp in [0.1, 1, 10]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"SimulatedAnnealing": {"start_temp": start_temp}},
        )


def test_StochasticTunnelingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="StochasticTunneling")

    for start_temp in [0.1, 1, 10]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"StochasticTunneling": {"start_temp": start_temp}},
        )


def test_ParallelTemperingOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="ParallelTempering")

    for n_swaps in [1, 10, 30]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"ParallelTempering": {"n_swaps": n_swaps}},
        )


def test_ParticleSwarmOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="ParticleSwarm")

    for n_particles in [2, 10, 30]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"ParticleSwarm": {"n_particles": n_particles}},
        )


def test_EvolutionStrategyOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=n_iter, optimizer="EvolutionStrategy")

    for individuals in [2, 10, 30]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"EvolutionStrategy": {"individuals": individuals}},
        )


def test_BayesianOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=int(n_iter / 33), optimizer="Bayesian")

    for warm_start_smbo in [True]:
        opt = Hyperactive(X, y, memory="long")
        opt.search(
            search_config,
            n_iter=int(n_iter / 33),
            optimizer={"Bayesian": {"warm_start_smbo": warm_start_smbo}},
        )


def test_TPE():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=int(n_iter / 5), optimizer="TPE")


def test_DecisionTreeOptimizer():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config, n_iter=int(n_iter / 33), optimizer="DecisionTree")

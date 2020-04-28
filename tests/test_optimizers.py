# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

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

n_iter = 100


def get_score(pos_new):
    return -(pos_new[0] * pos_new[0])


space_dim = np.array([10])
init_positions = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]


def _base_test(opt):
    for nth_init in range(len(init_positions)):
        pos_new = opt.init_pos(nth_init)
        score_new = get_score(pos_new)
        opt.evaluate(score_new)

    for nth_iter in range(len(init_positions), n_iter):
        pos_new = opt.iterate(nth_iter)
        score_new = get_score(pos_new)
        opt.evaluate(score_new)


def test_HillClimbingOptimizer():
    opt = HillClimbingOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_StochasticHillClimbingOptimizer():
    opt = StochasticHillClimbingOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_TabuOptimizer():
    opt = TabuOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_RandomSearchOptimizer():
    opt = RandomSearchOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_RandomRestartHillClimbingOptimizer():
    opt = RandomRestartHillClimbingOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_RandomAnnealingOptimizer():
    opt = RandomAnnealingOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_SimulatedAnnealingOptimizer():
    opt = SimulatedAnnealingOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_StochasticTunnelingOptimizer():
    opt = StochasticTunnelingOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_ParallelTemperingOptimizer():
    opt = ParallelTemperingOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_ParticleSwarmOptimizer():
    opt = ParticleSwarmOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_EvolutionStrategyOptimizer():
    opt = EvolutionStrategyOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_BayesianOptimizer():
    opt = BayesianOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_TreeStructuredParzenEstimators():
    opt = TreeStructuredParzenEstimators(init_positions, space_dim, opt_para={})
    _base_test(opt)


def test_DecisionTreeOptimizer():
    opt = DecisionTreeOptimizer(init_positions, space_dim, opt_para={})
    _base_test(opt)

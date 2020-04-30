# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import StochasticHillClimbingOptimizer

n_iter = 100


def get_score(pos_new):
    return -(pos_new[0] * pos_new[0])


space_dim = np.array([10])
init_positions = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]


def _base_test(opt, init_positions):
    for nth_init in range(len(init_positions)):
        pos_new = opt.init_pos(nth_init)
        score_new = get_score(pos_new)
        opt.evaluate(score_new)

    for nth_iter in range(len(init_positions), n_iter):
        pos_new = opt.iterate(nth_iter)
        score_new = get_score(pos_new)
        opt.evaluate(score_new)


def _test_StochasticHillClimbingOptimizer(
    init_positions=init_positions, space_dim=space_dim, opt_para={}
):
    opt = StochasticHillClimbingOptimizer(init_positions, space_dim, opt_para)
    _base_test(opt, init_positions)


def test_p_down():
    for p_down in [0.0001, 100]:
        opt_para = {"p_down": p_down}
        _test_StochasticHillClimbingOptimizer(opt_para=opt_para)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import ParallelTemperingOptimizer

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


def _test_ParallelTemperingOptimizer(
    init_positions=init_positions, space_dim=space_dim, opt_para={}
):
    opt = ParallelTemperingOptimizer(init_positions, space_dim, opt_para)
    _base_test(opt, init_positions)


def test_n_swaps():
    for n_iter_swap in [1, 3, 10, 33]:
        opt_para = {"n_iter_swap": n_iter_swap}
        _test_ParallelTemperingOptimizer(opt_para=opt_para)

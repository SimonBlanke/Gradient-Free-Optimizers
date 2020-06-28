# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import DecisionTreeOptimizer

n_iter = 33


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


def _test_DecisionTreeOptimizer(opt_para):
    opt = DecisionTreeOptimizer(init_positions, space_dim, opt_para=opt_para)
    _base_test(opt)


def test_start_up_evals():
    for start_up_evals in [1, 100]:
        opt_para = {"start_up_evals": start_up_evals}
        _test_DecisionTreeOptimizer(opt_para)


def test_warm_start_smbo():
    gpr_X, gpr_y = [], []
    for _ in range(10):
        pos = np.random.randint(0, high=9)
        pos = np.array([pos])
        gpr_X.append(pos)
        gpr_y.append(get_score(pos))

    for warm_start_smbo in [None, (gpr_X, gpr_y)]:
        opt_para = {"warm_start_smbo": warm_start_smbo}
        _test_DecisionTreeOptimizer(opt_para)


test_warm_start_smbo()


def test_max_sample_size():
    for max_sample_size in [10, 100, 10000, 10000000000]:
        opt_para = {"max_sample_size": max_sample_size}
        _test_DecisionTreeOptimizer(opt_para)


"""
def test_gpr():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(
        search_config,
        n_iter=n_iter,
        optimizer={"DecisionTree": {"tree_regressor": "random_forest"}},
    )
"""

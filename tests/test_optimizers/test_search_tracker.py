import pytest
from tqdm import tqdm
import numpy as np

from ._parametrize import optimizers


n_iter_para = ("n_iter", [(10), (20), (30)])


@pytest.mark.parametrize(*n_iter_para)
@pytest.mark.parametrize(*optimizers)
def test_search_tracker(Optimizer, n_iter):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-10, 11, 1)}
    initialize = {"vertices": 1}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(
        objective_function,
        n_iter=n_iter,
        memory=False,
        verbosity=False,
    )

    n_new_positions = 0
    n_new_scores = 0

    n_current_positions = 0
    n_current_scores = 0

    n_best_positions = 0
    n_best_scores = 0

    optimizers = opt.optimizers
    for optimizer in optimizers:
        n_new_positions = n_new_positions + len(optimizer.pos_new_list)
        n_new_scores = n_new_scores + len(optimizer.score_new_list)

        n_current_positions = n_current_positions + len(optimizer.pos_current_list)
        n_current_scores = n_current_scores + len(optimizer.score_current_list)

        n_best_positions = n_best_positions + len(optimizer.pos_best_list)
        n_best_scores = n_best_scores + len(optimizer.score_best_list)

    assert n_new_positions == n_iter
    assert n_new_scores == n_iter

    assert n_current_positions == n_current_scores
    assert n_current_positions <= n_new_positions

    assert n_best_positions == n_best_scores
    assert n_best_positions <= n_new_positions

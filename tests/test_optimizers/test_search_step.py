import numpy as np
import pytest

from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(-10, 10, 0.1),
}


@pytest.mark.parametrize(*optimizers)
def test_search_step_0(Optimizer):
    n_iter = 100

    opt = Optimizer(search_space)

    opt.init_search(
        objective_function,
        n_iter,
        max_time=None,
        max_score=None,
        early_stopping=None,
        memory=True,
        memory_warm_start=None,
        verbosity=["progress_bar", "print_results", "print_times"],
    )

    for nth_iter in range(n_iter):
        opt.search_step(nth_iter)

    opt.finish_search()

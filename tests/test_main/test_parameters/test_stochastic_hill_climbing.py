# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import StochasticHillClimbingOptimizer

n_iter = 1000


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 10, 1),
}


def test_p_accept():
    p_accept_low = 0.5
    p_accept_high = 1

    epsilon = 1 / np.inf

    opt = StochasticHillClimbingOptimizer(
        search_space,
        p_accept=p_accept_low,
        epsilon=epsilon,
        initialize={"random": 1},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_low = opt.n_transitions

    opt = StochasticHillClimbingOptimizer(
        search_space,
        p_accept=p_accept_high,
        epsilon=epsilon,
        initialize={"random": 1},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_high = opt.n_transitions

    print("\n n_transitions_low", n_transitions_low)
    print("\n n_transitions_high", n_transitions_high)

    lower_bound = int(n_iter * p_accept_low)
    lower_bound -= lower_bound * 0.1
    higher_bound = n_iter

    assert lower_bound < n_transitions_low < n_transitions_high < higher_bound

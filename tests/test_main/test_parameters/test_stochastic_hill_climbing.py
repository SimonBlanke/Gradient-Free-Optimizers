"""Tests for StochasticHillClimbingOptimizer p_accept parameter behavior."""
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


def test_step_size_parameter():
    # ensure the new argument is accepted and stored
    opt = StochasticHillClimbingOptimizer(search_space, step_size=0.5)
    assert hasattr(opt, "step_size")
    assert opt.step_size == 0.5


def test_step_size_conversion():
    # check internal epsilon values for discrete and continuous dims
    space_disc = {"x1": np.arange(0, 5, 1)}
    opt = StochasticHillClimbingOptimizer(space_disc, step_size=2)
    expected = np.array([2 / (len(space_disc["x1"]) - 1)])
    assert np.allclose(opt.epsilon_disc, expected)
    space_cont = {"x1": (0, 10)}
    opt2 = StochasticHillClimbingOptimizer(space_cont, step_size=2)
    assert np.allclose(opt2.epsilon_cont, np.array([0.2]))

"""Tests for SimulatedAnnealingOptimizer temperature and annealing parameters."""
# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import SimulatedAnnealingOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 10, 1),
}


n_iter = 1000


def test_start_temp_0():
    n_initialize = 1

    start_temp_0 = 0
    start_temp_1 = 0.1
    start_temp_10 = 1
    start_temp_100 = 100
    start_temp_inf = np.inf

    epsilon = 1 / np.inf

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_0,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_0 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_1,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_1 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_10,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_10 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_100,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_100 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_inf,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_inf = opt.n_transitions

    print("\n n_transitions_0", n_transitions_0)
    print("\n n_transitions_1", n_transitions_1)
    print("\n n_transitions_10", n_transitions_10)
    print("\n n_transitions_100", n_transitions_100)
    print("\n n_transitions_inf", n_transitions_inf)

    assert n_transitions_0 == start_temp_0
    assert (
        n_transitions_1
        == n_transitions_10
        == n_transitions_100
        == n_transitions_inf
        == n_iter - n_initialize
    )


def test_start_temp_1():
    n_initialize = 1

    start_temp_0 = 0
    start_temp_1 = 0.001
    start_temp_100 = 10000

    epsilon = 0.03

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_0,
        epsilon=epsilon,
        initialize={"random": n_initialize},
        random_state=42,
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_0 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_1,
        epsilon=epsilon,
        initialize={"random": n_initialize},
        random_state=100,
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_1 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        start_temp=start_temp_100,
        epsilon=epsilon,
        initialize={"random": n_initialize},
        random_state=100,
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_100 = opt.n_transitions

    print("\n n_transitions_0", n_transitions_0)
    print("\n n_transitions_1", n_transitions_1)
    print("\n n_transitions_100", n_transitions_100)

    assert n_transitions_0 == start_temp_0
    assert n_transitions_1 <= n_transitions_100


def test_annealing_rate_0():
    n_initialize = 1

    annealing_rate_0 = 0
    annealing_rate_1 = 0.1
    annealing_rate_100 = 0.99

    epsilon = 0.03

    opt = SimulatedAnnealingOptimizer(
        search_space,
        annealing_rate=annealing_rate_0,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_0 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        annealing_rate=annealing_rate_1,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_1 = opt.n_transitions

    opt = SimulatedAnnealingOptimizer(
        search_space,
        annealing_rate=annealing_rate_100,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_100 = opt.n_transitions

    print("\n n_transitions_0", n_transitions_0)
    print("\n n_transitions_1", n_transitions_1)
    print("\n n_transitions_100", n_transitions_100)

    assert n_transitions_0 in [0, 1]
    assert n_transitions_1 < n_transitions_100


def test_step_size_parameter():
    # ensure wrapper accepts step_size and converts epsilon
    space = {"x": (0, 10), "y": (0, 20)}
    opt = SimulatedAnnealingOptimizer(space, step_size=2)
    assert hasattr(opt, "step_size") and opt.step_size == 2
    # epsilon_cont should be computed accordingly
    assert np.allclose(opt.epsilon_cont, np.array([0.2, 0.1]))


def test_step_size_conversion():
    # verify epsilon_cont/disc for step_size
    space_disc = {"x": np.arange(0, 5, 1)}
    opt = SimulatedAnnealingOptimizer(space_disc, step_size=2)
    expected = np.array([2 / (len(space_disc["x"]) - 1)])
    assert np.allclose(opt.epsilon_disc, expected)
    space_cont = {"x": (0, 5)}
    opt2 = SimulatedAnnealingOptimizer(space_cont, step_size=2)
    assert np.allclose(opt2.epsilon_cont, np.array([0.4]))

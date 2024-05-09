# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from surfaces.test_functions.mathematical import SphereFunction

from gradient_free_optimizers import SimulatedAnnealingOptimizer


sphere_function = SphereFunction(n_dim=2)
objective_function = sphere_function.objective_function
search_space = sphere_function.search_space()


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
    start_temp_1 = 0.1
    start_temp_10 = 1
    start_temp_100 = 100
    start_temp_inf = np.inf

    epsilon = 0.03

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
    assert n_transitions_1 < n_transitions_10 < n_transitions_100 < n_transitions_inf


def test_annealing_rate_0():
    n_initialize = 1

    annealing_rate_0 = 0
    annealing_rate_1 = 0.1
    annealing_rate_10 = 0.5
    annealing_rate_100 = 0.9

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
        annealing_rate=annealing_rate_10,
        epsilon=epsilon,
        initialize={"random": n_initialize},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_10 = opt.n_transitions

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
    print("\n n_transitions_10", n_transitions_10)
    print("\n n_transitions_100", n_transitions_100)

    assert n_transitions_0 in [0, 1]
    # assert n_transitions_1 < n_transitions_10 < n_transitions_100

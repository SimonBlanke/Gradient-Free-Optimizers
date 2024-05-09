# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from surfaces.test_functions.mathematical import SphereFunction

from gradient_free_optimizers import StochasticHillClimbingOptimizer


sphere_function = SphereFunction(n_dim=2)
objective_function = sphere_function.objective_function
search_space = sphere_function.search_space()


n_iter_para = (
    "n_iter",
    [
        (300),
        (500),
        (1000),
    ],
)


@pytest.mark.parametrize(*n_iter_para)
def test_p_accept(n_iter):
    p_accept_low = 0.5
    p_accept_mid = 0.75
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
        p_accept=p_accept_mid,
        epsilon=epsilon,
        initialize={"random": 1},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_mid = opt.n_transitions

    opt = StochasticHillClimbingOptimizer(
        search_space,
        p_accept=p_accept_high,
        epsilon=epsilon,
        initialize={"random": 1},
    )
    opt.search(objective_function, n_iter=n_iter)
    n_transitions_high = opt.n_transitions

    print("\n n_transitions_low", n_transitions_low)
    print("\n n_transitions_mid", n_transitions_mid)
    print("\n n_transitions_high", n_transitions_high)

    lower_bound = int(n_iter * p_accept_low)
    lower_bound -= lower_bound * 0.1
    higher_bound = n_iter

    assert (
        lower_bound
        < n_transitions_low
        < n_transitions_mid
        < n_transitions_high
        < higher_bound
    )

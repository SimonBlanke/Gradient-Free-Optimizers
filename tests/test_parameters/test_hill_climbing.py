# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np


from gradient_free_optimizers import HillClimbingOptimizer


def parabola_function(para):
    loss = para["x"] * para["x"] + para["y"] * para["y"]
    return -loss


search_space = {
    "x": np.arange(-10, 11, 1),
    "y": np.arange(-10, 11, 1),
}


def test_epsilon_0():
    epsilon = 1 / np.inf

    opt = HillClimbingOptimizer(
        search_space, initialize={"vertices": 1}, epsilon=epsilon
    )
    opt.search(parabola_function, n_iter=100)

    search_data = opt.search_data
    scores = search_data["score"].values

    assert np.all(scores == -200)

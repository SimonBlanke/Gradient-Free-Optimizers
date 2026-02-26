"""Tests for HillClimbingOptimizer epsilon parameter behavior."""
# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

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
    opt.search(parabola_function, n_iter=100, verbosity=False)

    search_data = opt.search_data
    scores = search_data["score"].values

    assert np.all(scores == -200)


def test_step_size_0():
    # step_size of zero should mimic epsilon==0 behaviour
    step_size = 0.0
    opt = HillClimbingOptimizer(
        search_space, initialize={"vertices": 1}, step_size=step_size
    )
    opt.search(parabola_function, n_iter=100, verbosity=False)
    search_data = opt.search_data
    scores = search_data["score"].values
    assert np.all(scores == -200)


def test_step_size_convergence():
    # replicate issue: decreasing step_size should produce equal or better
    # final score on a fine grid
    space = {
        "x": np.arange(-10, 10, 0.01),
        "y": np.arange(-10, 10, 0.01),
    }

    def sphere(para):
        x, y = para["x"], para["y"]
        return -(x ** 2 + y ** 2)

    scores = []
    for step in [1, 0.5, 0.1, 0.01]:
        opt = HillClimbingOptimizer(space, step_size=step, random_state=42)
        opt.search(sphere, n_iter=500, verbosity=False)
        scores.append(opt.best_score)

    # smaller step size should achieve a score at least as good as largest
    assert scores[-1] >= max(scores[:-1])


def test_step_size_epsilon_conversion():
    # discrete dims should compute epsilon_disc such that sigma == step_size
    space_disc = {"x": np.arange(0, 10, 1), "y": np.arange(0, 20, 1)}
    step = 2
    opt = HillClimbingOptimizer(space_disc, step_size=step)
    expected = np.array([step / (len(space_disc["x"]) - 1), step / (len(space_disc["y"]) - 1)])
    assert np.allclose(opt.epsilon_disc, expected)

    # continuous dims should compute epsilon_cont similarly
    space_cont = {"x": (0, 10), "y": (0, 20)}
    opt2 = HillClimbingOptimizer(space_cont, step_size=step)
    expected2 = np.array([step / 10, step / 20])
    assert np.allclose(opt2.epsilon_cont, expected2)

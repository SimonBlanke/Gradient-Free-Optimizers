# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np


from gradient_free_optimizers import BayesianOptimizer


def parabola_function(para):
    loss = para["x"] * para["x"] + para["y"] * para["y"]
    return -loss


search_space = {
    "x": np.arange(-1, 1, 1),
    "y": np.arange(-1, 1, 1),
}


def test_replacement_0():
    opt = BayesianOptimizer(search_space, replacement=True)
    opt.search(parabola_function, n_iter=15)

    with pytest.raises(ValueError):
        opt = BayesianOptimizer(search_space, replacement=False)
        opt.search(parabola_function, n_iter=15)

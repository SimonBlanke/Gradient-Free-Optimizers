# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np


from gradient_free_optimizers import GridSearchOptimizer


def parabola_function(para):
    loss = para["x"] * para["x"] + para["y"] * para["y"]
    return -loss


search_space = {
    "x": np.arange(-100, 100, 1),
    "y": np.arange(-100, 100, 1),
}

n_iter = 50


def test_direction_0():
    n_initialize = 1
    opt = GridSearchOptimizer(
        search_space, initialize={"vertices": n_initialize}, direction="orthogonal"
    )
    opt.search(parabola_function, n_iter=n_iter)
    search_data = opt.search_data

    print("\n search_data \n", search_data, "\n")
    y_data_count = search_data["y"].value_counts().to_dict()
    print("\n y_data_count \n", y_data_count, "\n")

    assert y_data_count[-100] >= n_iter - n_initialize

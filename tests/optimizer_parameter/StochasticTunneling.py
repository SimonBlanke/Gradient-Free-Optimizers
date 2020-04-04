# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from hyperactive import Hyperactive

X, y = np.array([0]), np.array([0])
memory = False
n_iter = 25


def sphere_function(para, X_train, y_train):
    loss = []
    for key in para.keys():
        if key == "iteration":
            continue
        loss.append(para[key] * para[key])

    return -np.array(loss).sum()


search_config = {
    sphere_function: {
        "x1": list(np.arange(-3, 3, 0.1)),
        "x2": list(np.arange(-3, 3, 0.1)),
    }
}


def test_annealing_rate():
    for annealing_rate in [1, 0.1]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"StochasticTunneling": {"annealing_rate": annealing_rate}},
        )


def test_start_temp():
    for start_temp in [0.001, 10000]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"StochasticTunneling": {"start_temp": start_temp}},
        )

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


def test_skip_retrain():
    for skip_retrain in ["many", "some", "few", "never"]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"Bayesian": {"skip_retrain": skip_retrain}},
        )


def test_start_up_evals():
    for start_up_evals in [1, 100]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"Bayesian": {"start_up_evals": start_up_evals}},
        )


def test_warm_start_smbo():
    opt = Hyperactive(X, y, memory="long")
    opt.search(
        search_config, n_iter=n_iter, optimizer={"Bayesian": {"warm_start_smbo": True}}
    )


def test_max_sample_size():
    for max_sample_size in [10, 100, 10000, 10000000000]:
        opt = Hyperactive(X, y, memory=memory)
        opt.search(
            search_config,
            n_iter=n_iter,
            optimizer={"Bayesian": {"max_sample_size": True}},
        )


def test_gpr():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(
        search_config, n_iter=n_iter, optimizer={"Bayesian": {"gpr": "gp_linear"}}
    )

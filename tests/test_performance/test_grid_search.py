import pytest
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import reduce

from gradient_free_optimizers import GridSearchOptimizer

from surfaces.test_functions import SphereFunction, RastriginFunction


obj_func_l = (
    "objective_function",
    [
        (SphereFunction(n_dim=1, metric="score")),
        (RastriginFunction(n_dim=1, metric="score")),
    ],
)


@pytest.mark.parametrize(*obj_func_l)
def test_global_perf_0(objective_function):
    search_space = {"x0": np.arange(-10, 10, 0.1)}
    initialize = {"vertices": 2}

    print(
        "\n np.array(search_space.values()) \n",
        np.array(search_space.values()),
        np.array(search_space.values()).shape,
    )

    dim_sizes_list = [len(array) for array in search_space.values()]
    ss_size = reduce((lambda x, y: x * y), dim_sizes_list)

    n_opts = 10
    n_iter = ss_size

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = GridSearchOptimizer(
            search_space, initialize=initialize, random_state=rnd_st
        )
        opt.search(
            objective_function,
            n_iter=n_iter,
            memory=False,
            verbosity=False,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    print("\n score_mean", score_mean)
    print("\n n_iter", n_iter)

    assert score_mean > -0.001


obj_func_l = (
    "objective_function",
    [
        (SphereFunction(n_dim=2, metric="score")),
        (RastriginFunction(n_dim=2, metric="score")),
    ],
)


@pytest.mark.parametrize(*obj_func_l)
def test_global_perf_1(objective_function):
    search_space = {
        "x0": np.arange(-2, 1, 0.1),
        "x1": np.arange(-1, 2, 0.1),
    }
    initialize = {"vertices": 2}

    dim_sizes_list = [len(array) for array in search_space.values()]
    ss_size = reduce((lambda x, y: x * y), dim_sizes_list)

    n_opts = 10
    n_iter = ss_size

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = GridSearchOptimizer(
            search_space, initialize=initialize, random_state=rnd_st
        )
        opt.search(
            objective_function,
            n_iter=n_iter,
            memory=False,
            verbosity=False,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    print("\n score_mean", score_mean)

    assert score_mean > -0.001

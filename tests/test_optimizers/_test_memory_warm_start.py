import pytest
import time
import numpy as np
import pandas as pd
from gradient_free_optimizers import GridSearchOptimizer


from ._parametrize import optimizers_non_smbo, optimizers_smbo


@pytest.mark.parametrize(*optimizers_non_smbo)
def test_memory_warm_start_0(Optimizer_non_smbo):
    def objective_function(para):
        time.sleep(0.1)
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    n_iter = 20

    c_time1 = time.time()
    opt0 = GridSearchOptimizer(search_space)
    opt0.search(objective_function, n_iter=n_iter)
    diff_time1 = time.time() - c_time1

    c_time2 = time.time()
    opt1 = Optimizer_non_smbo(search_space)
    opt1.search(objective_function, n_iter=n_iter, memory_warm_start=opt0.search_data)
    diff_time2 = time.time() - c_time2

    print("\n diff_time1 ", diff_time1)
    print("\n diff_time2 ", diff_time2)

    assert diff_time2 < diff_time1 * 0.5


@pytest.mark.parametrize(*optimizers_smbo)
def test_memory_warm_start_1(Optimizer_smbo):
    def objective_function(para):
        time.sleep(0.5)
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 1, 1),
    }

    n_iter = 2

    c_time1 = time.time()
    opt0 = GridSearchOptimizer(search_space)
    opt0.search(objective_function, n_iter=n_iter)
    diff_time1 = time.time() - c_time1

    c_time2 = time.time()
    opt1 = Optimizer_smbo(search_space)
    opt1.search(objective_function, n_iter=n_iter, memory_warm_start=opt0.search_data)
    diff_time2 = time.time() - c_time2

    print("\n diff_time1 ", diff_time1)
    print("\n diff_time2 ", diff_time2)

    assert diff_time2 < diff_time1 * 0.5

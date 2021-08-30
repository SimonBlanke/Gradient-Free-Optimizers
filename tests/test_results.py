import numpy as np
import pandas as pd
from gradient_free_optimizers import RandomSearchOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100000, 0.1),
}


def test_attributes_results_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert isinstance(opt.search_data, pd.DataFrame)


def test_attributes_results_1():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert set(search_space.keys()) < set(opt.search_data.columns)


def test_attributes_results_2():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert "x1" in list(opt.search_data.columns)


def test_attributes_results_3():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)

    assert "score" in list(opt.search_data.columns)


def test_attributes_results_4():
    opt = RandomSearchOptimizer(search_space, initialize={"warm_start": [{"x1": 0}]})
    opt.search(objective_function, n_iter=1)

    assert 0 in list(opt.search_data["x1"].values)


def test_attributes_results_5():
    opt = RandomSearchOptimizer(search_space, initialize={"warm_start": [{"x1": 10}]})
    opt.search(objective_function, n_iter=1)

    assert 10 in list(opt.search_data["x1"].values)


def test_attributes_results_6():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    opt = RandomSearchOptimizer(search_space, initialize={"random": 1})
    opt.search(objective_function, n_iter=20, memory=False)

    x1_results = list(opt.search_data["x1"].values)

    print("\n x1_results \n", x1_results)

    assert len(set(x1_results)) < len(x1_results)


"""
def test_attributes_results_7():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function, n_iter=20, initialize={"random": 1}, memory=True
    )

    x1_results = list(opt.search_data["x1"].values)

    print("\n x1_results \n", x1_results)

    assert len(set(x1_results)) == len(x1_results)


def test_attributes_results_8():
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    results = pd.DataFrame(np.arange(-10, 10, 1), columns=["x1"])
    results["score"] = 0

    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function,
        n_iter=100,
        initialize={},
        memory=True,
        memory_warm_start=results,
    )

    print("\n opt.search_data \n", opt.search_data)

    x1_results = list(opt.search_data["x1"].values)

    assert 10 == x1_results[0]
"""

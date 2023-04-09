import numpy as np
import pytest

from ._parametrize import optimizers


@pytest.mark.parametrize(*optimizers)
def test_constr_opt_0(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(-15, 15, 1),
    }

    def constraint_1(para):
        return para["x1"] > -5

    constraints_list = [constraint_1]

    opt = Optimizer(search_space, constraints=constraints_list)
    opt.search(objective_function, n_iter=20)

    search_data = opt.search_data
    x0_values = search_data["x1"].values

    print("\n search_data \n", search_data, "\n")

    assert np.all(x0_values > -5)


@pytest.mark.parametrize(*optimizers)
def test_constr_opt_1(Optimizer):
    def objective_function(para):
        score = -(para["x1"] * para["x1"] + para["x2"] * para["x2"])
        return score

    search_space = {
        "x1": np.arange(-10, 10, 1),
        "x2": np.arange(-10, 10, 1),
    }

    def constraint_1(para):
        return para["x1"] > -5

    constraints_list = [constraint_1]

    opt = Optimizer(search_space, constraints=constraints_list)
    opt.search(objective_function, n_iter=50)

    search_data = opt.search_data
    x0_values = search_data["x1"].values

    print("\n search_data \n", search_data, "\n")

    assert np.all(x0_values > -5)


@pytest.mark.parametrize(*optimizers)
def test_constr_opt_2(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {
        "x1": np.arange(-10, 10, 0.1),
    }

    def constraint_1(para):
        return para["x1"] > -5

    def constraint_2(para):
        return para["x1"] < 5

    constraints_list = [constraint_1, constraint_2]

    opt = Optimizer(search_space, constraints=constraints_list)
    opt.search(objective_function, n_iter=50)

    search_data = opt.search_data
    x0_values = search_data["x1"].values

    print("\n search_data \n", search_data, "\n")

    assert np.all(x0_values > -5)
    assert np.all(x0_values < 5)

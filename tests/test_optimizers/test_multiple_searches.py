import pytest
import numpy as np

from ._parametrize import optimizers_singleOpt, optimizers_PopBased, optimizers_SBOM


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 1, 1)}


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_searches_0(Optimizer):

    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=1)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 2

    assert opt.n_init_total == 1
    assert opt.n_iter_total == 1
    assert opt.n_init_search == 0
    assert opt.n_iter_search == 1


@pytest.mark.parametrize(*optimizers_PopBased)
def test_searches_pop_0(Optimizer):

    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=1)

    print("\n opt.search_data \n", opt.search_data)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 2

    assert opt.n_init_total == 2
    assert opt.n_iter_total == 0
    assert opt.n_init_search == 1
    assert opt.n_iter_search == 0


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_searches_1(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)

    print("\n opt.search_data \n", opt.search_data)

    opt.search(objective_function, n_iter=10)

    print("\n opt.search_data \n", opt.search_data)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 11

    assert opt.n_init_total == 1
    assert opt.n_iter_total == 10
    assert opt.n_init_search == 0
    assert opt.n_iter_search == 10


@pytest.mark.parametrize(*optimizers_PopBased)
def test_searches_pop_1(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=10)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 11

    assert opt.n_init_total != 1
    assert opt.n_iter_total != 10
    assert opt.n_init_search != 0
    assert opt.n_iter_search != 10


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_searches_2(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=20)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 21

    assert opt.n_init_total == 1
    assert opt.n_iter_total == 20
    assert opt.n_init_search == 0
    assert opt.n_iter_search == 20


@pytest.mark.parametrize(*optimizers_PopBased)
def test_searches_pop_2(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=1)
    opt.search(objective_function, n_iter=20)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 21

    assert opt.n_init_total != 1
    assert opt.n_iter_total != 20
    assert opt.n_init_search != 0
    assert opt.n_iter_search != 20


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_searches_3(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=20)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 30

    assert opt.n_init_total == 1
    assert opt.n_iter_total == 29
    assert opt.n_init_search == 0
    assert opt.n_iter_search == 20


@pytest.mark.parametrize(*optimizers_PopBased)
def test_searches_pop_3(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=20)
    opt.search(objective_function, n_iter=20)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 40

    assert 1 < opt.n_init_total < 20
    assert opt.n_iter_total > 20
    assert opt.n_init_search == 0
    assert opt.n_iter_search == 20


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_searches_4(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=10)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 30

    assert opt.n_init_total == 1
    assert opt.n_iter_total == 29
    assert opt.n_init_search == 0
    assert opt.n_iter_search == 10


@pytest.mark.parametrize(*optimizers_PopBased)
def test_searches_pop_4(Optimizer):
    initialize = {"warm_start": [{"x1": -100}]}

    opt = Optimizer(search_space, initialize=initialize)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=10)
    opt.search(objective_function, n_iter=10)

    assert -100 in opt.search_data["x1"].values
    assert len(opt.search_data["x1"]) == 30

    assert opt.n_init_total != 1
    assert opt.n_iter_total != 29
    assert opt.n_init_search == 0
    assert opt.n_iter_search == 10

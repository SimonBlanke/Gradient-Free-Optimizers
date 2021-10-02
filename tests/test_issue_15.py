import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


""" --- test search spaces with mixed int/float types --- """


def test_mixed_type_search_space_0():
    def objective_function(para):
        assert isinstance(para["x1"], int)

        return 1

    search_space = {
        "x1": range(10, 20),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=10000)


def test_mixed_type_search_space_1():
    def objective_function(para):
        assert isinstance(para["x2"], float)

        return 1

    search_space = {
        "x2": np.arange(1, 2, 0.1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=10000)


def test_mixed_type_search_space_2():
    def objective_function(para):
        assert isinstance(para["x1"], int)
        assert isinstance(para["x2"], float)

        return 1

    search_space = {
        "x1": range(10, 20),
        "x2": np.arange(1, 2, 0.1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=10000)

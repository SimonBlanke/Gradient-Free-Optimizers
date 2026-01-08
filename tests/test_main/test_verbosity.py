import numpy as np

from gradient_free_optimizers import RandomSearchOptimizer


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100, 0.1),
}


def test_verbosity_0():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100, verbosity=False)


def test_verbosity_1():
    opt = RandomSearchOptimizer(search_space,)
    opt.search(
        objective_function,
        n_iter=100,
        verbosity=["progress_bar", "print_results", "print_times"],
    )


def test_verbosity_2():
    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function,
        n_iter=100,
        verbosity=["print_results", "print_times"],
    )


def test_verbosity_3():
    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function,
        n_iter=100,
        verbosity=["progress_bar", "print_times"],
    )


def test_verbosity_4():
    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function,
        n_iter=100,
        verbosity=["progress_bar", "print_results"],
    )


def test_verbosity_5():
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100, verbosity=[])


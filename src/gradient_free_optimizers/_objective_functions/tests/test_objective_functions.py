from gradient_free_optimizers import RandomSearchOptimizer


def test_ackley():
    from gradient_free_optimizers._objective_functions._ackley_function import (
        AckleyFunction,
    )

    ackley = AckleyFunction()

    opt = RandomSearchOptimizer(ackley.search_space)
    opt.search(ackley.objective_function, n_iter=100)


def test_ackley():
    from gradient_free_optimizers._objective_functions._sphere_function import (
        SphereFunction,
    )

    sphere = SphereFunction(n_dim=3)

    opt = RandomSearchOptimizer(sphere.search_space)
    opt.search(sphere.objective_function, n_iter=100)

import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


""" --- test search spaces with mixed int/float types --- """


def test_mixed_type_search_space():
    def objective_function(para):
        nonlocal para_types
        for v, t in zip(para.values(), para_types):
            assert isinstance(v, t)
        score = 0
        for x1 in range(para["x1"]):
            score += -(x1 ** 2) + para["x2"] + 100
        return score

    search_space = {
        "x1": range(10, 20),
        "x2": np.arange(1, 2, 0.1),
    }
    para_types = [int, float]
    expected_pos = [1, 9]

    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=10000)

    for best_para_val, expected_p, dim_space, p_type in zip(
        opt.best_para.values(), expected_pos, search_space.values(), para_types
    ):
        print("p_type", p_type)
        print("dim_space", dim_space)

        assert best_para_val == dim_space[expected_p]
        assert isinstance(best_para_val, p_type)

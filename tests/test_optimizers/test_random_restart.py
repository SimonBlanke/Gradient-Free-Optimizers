import pytest
import numpy as np

from ._parametrize import optimizers_local


@pytest.mark.parametrize(*optimizers_local)
def test_convex_convergence_singleOpt(Optimizer):
    def objective_function(para):
        score = -(para["x1"] * para["x1"])
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    init1 = {
        "x1": -1000,
    }
    initialize = {"warm_start": [init1]}

    n_opts = 33

    scores = []
    for rnd_st in range(n_opts):
        opt = Optimizer(search_space, rand_rest_p=1)
        opt.search(
            objective_function,
            n_iter=20,
            random_state=rnd_st,
            memory=False,
            verbosity=False,
            initialize=initialize,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    print("score_mean", score_mean)

    assert score_mean > -400


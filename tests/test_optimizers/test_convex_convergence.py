import pytest
from tqdm import tqdm
import numpy as np

from ._parametrize import (
    optimizers_singleOpt,
    optimizers_PopBased,
    optimizers_SBOM,
)


@pytest.mark.parametrize(*optimizers_singleOpt)
def test_convex_convergence_singleOpt(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-100, 101, 1)}
    initialize = {"vertices": 1}

    n_opts = 33

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = Optimizer(search_space)
        opt.search(
            objective_function,
            n_iter=100,
            random_state=rnd_st,
            memory=False,
            verbosity=False,
            initialize=initialize,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    assert score_mean > -25


@pytest.mark.parametrize(*optimizers_PopBased)
def test_convex_convergence_popBased(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-100, 101, 1)}
    initialize = {"vertices": 2, "grid": 2}

    n_opts = 33

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = Optimizer(search_space)
        opt.search(
            objective_function,
            n_iter=80,
            random_state=rnd_st,
            memory=False,
            verbosity=False,
            initialize=initialize,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    assert score_mean > -25


@pytest.mark.parametrize(*optimizers_SBOM)
def test_convex_convergence_SBOM(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-33, 33, 1)}
    initialize = {"vertices": 2}

    n_opts = 10

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = Optimizer(search_space)
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

    print("scores", scores)

    print("score_mean", score_mean)
    assert score_mean > -25
    # assert False


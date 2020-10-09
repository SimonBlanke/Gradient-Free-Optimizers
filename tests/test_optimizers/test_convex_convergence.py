import pytest
from tqdm import tqdm
import numpy as np

from ._parametrize import optimizers_noSBOM, optimizers_SBOM


@pytest.mark.parametrize(*optimizers_noSBOM)
def test_convex_convergence_noSBOM(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-33, 33, 1)}
    initialize = {"vertices": 2}

    n_opts = 33

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = Optimizer(search_space)
        opt.search(
            objective_function,
            n_iter=50,
            random_state=rnd_st,
            memory=False,
            verbosity={"print_results": False, "progress_bar": False},
            initialize=initialize,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()
    print("scores", scores)

    assert -500 < score_mean


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
            n_iter=30,
            random_state=rnd_st,
            memory=False,
            verbosity={"print_results": False, "progress_bar": False},
            initialize=initialize,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()
    print("scores", scores)

    assert -500 < score_mean


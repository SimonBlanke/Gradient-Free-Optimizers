import pytest
from tqdm import tqdm
import numpy as np

from ._parametrize import pytest_parameter


@pytest.mark.parametrize(*pytest_parameter)
def test_convex_convergence(Optimizer):
    def objective_function(pos_new):
        score = -pos_new[0] * pos_new[0]
        return score

    search_space = [np.arange(-100, 100, 1)]
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
            verbosity={"print_results": False, "progress_bar": False,},
            initialize=initialize,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    assert -500 < score_mean


import pytest
from tqdm import tqdm
import numpy as np

from surfaces.test_functions import RastriginFunction

from gradient_free_optimizers import (
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
)


opt_global_l = (
    "Optimizer",
    [
        (RandomSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
    ],
)


@pytest.mark.parametrize(*opt_global_l)
def test_global_perf(Optimizer):
    ackley_function = RastriginFunction(n_dim=1, metric="score")

    search_space = {"x0": np.arange(-100, 101, 1)}
    initialize = {"vertices": 2}

    n_opts = 33
    n_iter = 100

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = Optimizer(search_space, initialize=initialize, random_state=rnd_st)
        opt.search(
            ackley_function,
            n_iter=n_iter,
            memory=False,
            verbosity=False,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    print("\n score_mean", score_mean)

    assert score_mean > -5

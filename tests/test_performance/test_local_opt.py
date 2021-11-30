import pytest
from tqdm import tqdm
import numpy as np

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
)


opt_local_l = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
    ],
)


@pytest.mark.parametrize(*opt_local_l)
def test_local_perf(Optimizer):
    def objective_function(para):
        score = -para["x1"] * para["x1"]
        return score

    search_space = {"x1": np.arange(-100, 101, 1)}
    initialize = {"vertices": 2}

    n_opts = 33
    n_iter = 100

    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = Optimizer(search_space, initialize=initialize, random_state=rnd_st)
        opt.search(
            objective_function,
            n_iter=n_iter,
            memory=False,
            verbosity=False,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    print("\n score_mean", score_mean)

    assert score_mean > -5

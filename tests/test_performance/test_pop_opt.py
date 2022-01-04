import pytest
from tqdm import tqdm
import numpy as np

from surfaces.test_functions import SphereFunction, RastriginFunction

from gradient_free_optimizers import (
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    RandomSearchOptimizer,
)


opt_pop_l = (
    "Optimizer",
    [
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
    ],
)


obj_func_l = (
    "objective_function",
    [
        (SphereFunction(n_dim=2, metric="score")),
        (RastriginFunction(n_dim=2, metric="score")),
    ],
)


@pytest.mark.parametrize(*obj_func_l)
@pytest.mark.parametrize(*opt_pop_l)
def test_pop_perf_0(Optimizer, objective_function):
    search_space = {
        "x0": np.arange(-100, 101, 0.1),
        "x1": np.arange(-100, 101, 0.1),
    }
    initialize = {"vertices": 4, "random": 11}

    n_opts = 10
    n_iter = 1200

    scores = []
    scores_rnd = []
    for rnd_st in tqdm(range(n_opts)):
        opt = Optimizer(search_space, initialize=initialize, random_state=rnd_st)
        opt.search(
            objective_function,
            n_iter=n_iter,
            memory=False,
            verbosity=False,
        )

        opt_rnd = RandomSearchOptimizer(
            search_space, initialize=initialize, random_state=rnd_st
        )
        opt_rnd.search(
            objective_function,
            n_iter=n_iter,
            memory=False,
            verbosity=False,
        )

        scores.append(opt.best_score)
        scores_rnd.append(opt_rnd.best_score)

    score_mean = np.array(scores).mean()
    score_mean_rnd = np.array(scores_rnd).mean()

    print("\n score_mean", score_mean)
    print("\n score_mean_rnd", score_mean_rnd)

    score_norm = (score_mean_rnd - score_mean) / (score_mean_rnd + score_mean)
    print("\n score_norm", score_norm)

    assert score_norm > 0.20

import pytest
import numpy as np
from tqdm import tqdm

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    TabuOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    DecisionTreeOptimizer,
    EnsembleOptimizer,
)


def objective_function(pos_new):
    score = -pos_new[0] * pos_new[0]
    return score


search_space = [np.arange(-100, 100, 1)]
initialize = {"vertices": 2}

n_opts = 33
n_iter = 50
min_score_accept = -500


pytest_parameter = (
    "test_input",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (TabuOptimizer),
        (RandomSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (SimulatedAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (DecisionTreeOptimizer),
        (EnsembleOptimizer),
    ],
)


@pytest.mark.parametrize(*pytest_parameter)
def test_convex_convergence(test_input):
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = test_input(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            verbosity={"print_results": False, "progress_bar": False,},
            initialize=initialize,
        )

        scores.append(opt.best_score)
    score_mean = np.array(scores).mean()

    assert min_score_accept < score_mean


@pytest.mark.parametrize(*pytest_parameter)
def test_exploration(test_input):
    def objective_function(pos_new):
        score = -(pos_new[0] * pos_new[0] + pos_new[1] * pos_new[1])
        return score

    search_space = [np.arange(-20, 20, 1), np.arange(0, 3, 1)]
    init1 = [-20, 1]

    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function,
        n_iter=50,
        memory=False,
        verbosity={"print_results": False, "progress_bar": False,},
        initialize={"warm_start": [init1]},
    )

    uniques_2nd_dim = list(np.unique(opt.values[:, 1]))

    assert 0 in uniques_2nd_dim
    assert 2 in uniques_2nd_dim


"""Basic smoke tests for all optimizer algorithms."""

import numpy as np
import pytest

from gradient_free_optimizers import (
    BayesianOptimizer,
    DownhillSimplexOptimizer,
    EvolutionStrategyOptimizer,
    ForestOptimizer,
    GridSearchOptimizer,
    HillClimbingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomAnnealingOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticHillClimbingOptimizer,
    TreeStructuredParzenEstimators,
)

optimizers = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        (DownhillSimplexOptimizer),
        (RandomSearchOptimizer),
        (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (PowellsMethod),
        (PatternSearch),
        (RandomAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
    ],
)


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 10, 1),
}


@pytest.mark.parametrize(*optimizers)
def test_opt_algos_0(Optimizer):
    opt = Optimizer(search_space)
    opt.search(objective_function, n_iter=15)

    _ = opt.best_para
    _ = opt.best_score
    _ = opt.search_data

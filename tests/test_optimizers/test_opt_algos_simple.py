from gradient_free_optimizers.optimizers.core_optimizer import search_tracker
import pytest

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    PowellsMethod,
    PatternSearch,
    RandomAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)

from surfaces.test_functions import SphereFunction

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


sphere_function = SphereFunction(n_dim=2, metric="score")


@pytest.mark.parametrize(*optimizers)
def test_opt_algos_0(Optimizer):
    opt = Optimizer(sphere_function.search_space())
    opt.search(sphere_function, n_iter=15)

    _ = opt.best_para
    _ = opt.best_score
    _ = opt.search_data

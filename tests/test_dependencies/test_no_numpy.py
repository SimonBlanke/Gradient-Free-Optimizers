# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Dependency Isolation Tests - No numpy.

These tests verify that Gradient-Free-Optimizers works correctly
when numpy is NOT installed. All array operations fall back to the
pure Python GFOArray backend.

Test Configuration:
- initialize: {"random": 3} (minimal warm-up)
- n_iter: 10 (quick iteration count)
- Small discrete search space using backend linspace
"""

import os

import pytest

_ci_strict = os.environ.get("GFO_CI_STRICT")

try:
    import numpy

    if _ci_strict:
        raise RuntimeError(
            "numpy is installed but must be absent (GFO_CI_STRICT is set)"
        )
    pytest.skip(
        "numpy is installed - these tests require numpy to be absent",
        allow_module_level=True,
    )
except ImportError:
    pass

from gradient_free_optimizers import (
    BayesianOptimizer,
    DifferentialEvolutionOptimizer,
    DirectAlgorithm,
    DownhillSimplexOptimizer,
    EvolutionStrategyOptimizer,
    ForestOptimizer,
    GeneticAlgorithmOptimizer,
    GridSearchOptimizer,
    HillClimbingOptimizer,
    LipschitzOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomAnnealingOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    SpiralOptimization,
    StochasticHillClimbingOptimizer,
    TreeStructuredParzenEstimators,
)
from gradient_free_optimizers._array_backend import linspace

SEARCH_SPACE = {
    "x1": linspace(-5, 5, 11),
    "x2": linspace(-5, 5, 11),
}


def objective_function(para):
    return -(para["x1"] ** 2 + para["x2"] ** 2)


OPTIMIZERS = [
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
    LipschitzOptimizer,
    DirectAlgorithm,
    RandomAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
]


class TestNoNumpy:
    """Test all optimizers work without numpy installed."""

    @pytest.mark.parametrize("Optimizer", OPTIMIZERS)
    def test_optimizer_without_numpy(self, Optimizer):
        opt = Optimizer(
            SEARCH_SPACE,
            initialize={"random": 3},
            random_state=42,
        )
        opt.search(
            objective_function,
            n_iter=10,
            verbosity=False,
        )

        assert opt.best_para is not None
        assert opt.best_score is not None


def test_numpy_not_installed():
    """Verify numpy is not available in this test environment."""
    with pytest.raises(ImportError):
        import numpy  # noqa: F401

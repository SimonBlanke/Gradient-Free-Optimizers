# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Dependency Isolation Tests - No sklearn

These tests verify that Gradient-Free-Optimizers works correctly
when scikit-learn is NOT installed. This ensures the library can
function with minimal dependencies.

Test Configuration:
- initialize: {"random": 3} (minimal warm-up)
- n_iter: 10 (quick iteration count)
- Small numeric search space
"""

import sys

import numpy as np
import pytest

# Verify sklearn is NOT installed before running tests
try:
    import sklearn

    pytest.skip(
        "sklearn is installed - these tests require sklearn to be absent",
        allow_module_level=True,
    )
except ImportError:
    pass  # This is expected - sklearn should not be installed


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
)


# Small numeric search space for fast tests
SEARCH_SPACE = {
    "x1": np.linspace(-5, 5, 11),
    "x2": np.linspace(-5, 5, 11),
}


def objective_function(para):
    """Simple sphere function for testing."""
    return -(para["x1"] ** 2 + para["x2"] ** 2)


# All optimizers to test
ALL_OPTIMIZERS = [
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


class TestNoSklearn:
    """Test all optimizers work without sklearn installed."""

    @pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
    def test_optimizer_without_sklearn(self, Optimizer):
        """
        Test that optimizer runs successfully without sklearn.

        This test verifies:
        1. Optimizer can be instantiated
        2. Search runs without ImportError
        3. Results are accessible
        """
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

        # Verify results are accessible
        assert opt.best_para is not None
        assert opt.best_score is not None
        assert opt.search_data is not None
        assert len(opt.search_data) > 0


def test_sklearn_not_installed():
    """Explicit test that sklearn is not available."""
    with pytest.raises(ImportError):
        import sklearn  # noqa: F401

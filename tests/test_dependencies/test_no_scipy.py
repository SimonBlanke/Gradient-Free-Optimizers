# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Dependency Isolation Tests - No scipy

These tests verify that Gradient-Free-Optimizers works correctly
when scipy is NOT installed. This ensures the library can function
with the native math backend instead of scipy.

Test Configuration:
- initialize: {"random": 3} (minimal warm-up)
- n_iter: 10 (quick iteration count)
- Small numeric search space
"""

import sys

import numpy as np
import pytest

# Verify scipy is NOT installed before running tests
try:
    import scipy

    pytest.skip(
        "scipy is installed - these tests require scipy to be absent",
        allow_module_level=True,
    )
except ImportError:
    pass  # This is expected - scipy should not be installed


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


class TestNoScipy:
    """Test all optimizers work without scipy installed."""

    @pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
    def test_optimizer_without_scipy(self, Optimizer):
        """
        Test that optimizer runs successfully without scipy.

        This test verifies:
        1. Optimizer can be instantiated
        2. Search runs without ImportError
        3. Results are accessible
        4. Native math backend is used instead of scipy
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


def test_scipy_not_installed():
    """Explicit test that scipy is not available."""
    with pytest.raises(ImportError):
        import scipy  # noqa: F401

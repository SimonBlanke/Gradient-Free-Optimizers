"""API tests for extended search space support.

These tests ensure that all optimizers work correctly with the new
extended search space types:
- Continuous dimensions: (min, max) tuples
- Categorical dimensions: Python lists
- Mixed dimensions: combination of all three types

"""

import numpy as np
import pytest

from gradient_free_optimizers import (
    BayesianOptimizer,
    DownhillSimplexOptimizer,
    EvolutionStrategyOptimizer,
    ForestOptimizer,
    GeneticAlgorithmOptimizer,
    HillClimbingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticHillClimbingOptimizer,
)

# =============================================================================
# Search space fixtures
# =============================================================================


CONTINUOUS_SEARCH_SPACE = {
    "x": (-5.0, 5.0),
    "y": (0.0, 10.0),
}

CATEGORICAL_SEARCH_SPACE = {
    "optimizer": ["adam", "sgd", "rmsprop"],
    "use_bias": [True, False],
}

MIXED_SEARCH_SPACE = {
    "x": np.arange(-5, 5, 1),  # Discrete numerical
    "y": (-5.0, 5.0),  # Continuous
    "algorithm": ["adam", "sgd", "rmsprop"],  # Categorical
}


def simple_objective(params):
    """Simple objective function for testing."""
    score = 0
    for key, value in params.items():
        if isinstance(value, (int | float)):
            score -= value**2
        elif isinstance(value, str):
            if value == "adam":
                score += 0.5
        elif isinstance(value, bool):
            if value:
                score += 0.1
    return score


# =============================================================================
# Single-individual optimizers (should all work)
# =============================================================================


SINGLE_OPTIMIZERS = [
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
]

SMBO_OPTIMIZERS = [
    ForestOptimizer,
    BayesianOptimizer,
]


@pytest.mark.parametrize("Optimizer", SINGLE_OPTIMIZERS)
class TestSingleOptimizersContinuous:
    """Test single-individual optimizers with continuous search spaces."""

    def test_continuous_search_space(self, Optimizer):
        """Optimizer should accept and work with continuous dimensions."""
        opt = Optimizer(CONTINUOUS_SEARCH_SPACE, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert isinstance(opt.best_para["x"], float)
        assert isinstance(opt.best_para["y"], float)
        assert -5.0 <= opt.best_para["x"] <= 5.0
        assert 0.0 <= opt.best_para["y"] <= 10.0


@pytest.mark.parametrize("Optimizer", SINGLE_OPTIMIZERS)
class TestSingleOptimizersCategorical:
    """Test single-individual optimizers with categorical search spaces."""

    def test_categorical_search_space(self, Optimizer):
        """Optimizer should accept and work with categorical dimensions."""
        opt = Optimizer(CATEGORICAL_SEARCH_SPACE, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]
        assert opt.best_para["use_bias"] in [True, False]


@pytest.mark.parametrize("Optimizer", SINGLE_OPTIMIZERS)
class TestSingleOptimizersMixed:
    """Test single-individual optimizers with mixed search spaces."""

    def test_mixed_search_space(self, Optimizer):
        """Optimizer should accept and work with mixed dimension types."""
        opt = Optimizer(MIXED_SEARCH_SPACE, random_state=42)
        opt.search(simple_objective, n_iter=15, verbosity=[])

        assert opt.best_para is not None
        # Discrete numerical
        assert opt.best_para["x"] in list(range(-5, 5))
        # Continuous
        assert isinstance(opt.best_para["y"], float)
        assert -5.0 <= opt.best_para["y"] <= 5.0
        # Categorical
        assert opt.best_para["algorithm"] in ["adam", "sgd", "rmsprop"]


# =============================================================================
# SMBO optimizers (should all work)
# =============================================================================


@pytest.mark.parametrize("Optimizer", SMBO_OPTIMIZERS)
class TestSMBOOptimizersContinuous:
    """Test SMBO optimizers with continuous search spaces."""

    def test_continuous_search_space(self, Optimizer):
        """SMBO optimizer should work with continuous dimensions."""
        opt = Optimizer(CONTINUOUS_SEARCH_SPACE, random_state=42)
        opt.search(simple_objective, n_iter=15, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["x"] <= 5.0
        assert 0.0 <= opt.best_para["y"] <= 10.0


@pytest.mark.parametrize("Optimizer", SMBO_OPTIMIZERS)
class TestSMBOOptimizersCategorical:
    """Test SMBO optimizers with categorical search spaces."""

    def test_categorical_search_space(self, Optimizer):
        """SMBO optimizer should work with categorical dimensions."""
        opt = Optimizer(CATEGORICAL_SEARCH_SPACE, random_state=42)
        opt.search(simple_objective, n_iter=15, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]


@pytest.mark.parametrize("Optimizer", SMBO_OPTIMIZERS)
class TestSMBOOptimizersMixed:
    """Test SMBO optimizers with mixed search spaces."""

    def test_mixed_search_space(self, Optimizer):
        """SMBO optimizer should work with mixed dimension types."""
        opt = Optimizer(MIXED_SEARCH_SPACE, random_state=42)
        opt.search(simple_objective, n_iter=15, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["y"] <= 5.0
        assert opt.best_para["algorithm"] in ["adam", "sgd", "rmsprop"]


# =============================================================================
# Population-based optimizers
# =============================================================================

POPULATION_OPTIMIZERS = [
    ParticleSwarmOptimizer,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    ParallelTemperingOptimizer,
]


@pytest.mark.parametrize("Optimizer", POPULATION_OPTIMIZERS)
class TestPopulationOptimizersContinuous:
    """Test population-based optimizers with continuous search spaces."""

    def test_continuous_search_space(self, Optimizer):
        """Population optimizer should work with continuous dimensions."""
        opt = Optimizer(CONTINUOUS_SEARCH_SPACE, population=3, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["x"] <= 5.0


@pytest.mark.parametrize("Optimizer", POPULATION_OPTIMIZERS)
class TestPopulationOptimizersCategorical:
    """Test population-based optimizers with categorical search spaces."""

    def test_categorical_search_space(self, Optimizer):
        """Population optimizer should work with categorical dimensions."""
        opt = Optimizer(CATEGORICAL_SEARCH_SPACE, population=3, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]


@pytest.mark.parametrize("Optimizer", POPULATION_OPTIMIZERS)
class TestPopulationOptimizersMixed:
    """Test population-based optimizers with mixed search spaces."""

    def test_mixed_search_space(self, Optimizer):
        """Population optimizer should work with mixed dimension types."""
        opt = Optimizer(MIXED_SEARCH_SPACE, population=3, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["algorithm"] in ["adam", "sgd", "rmsprop"]


# =============================================================================
# Reproducibility tests
# =============================================================================


class TestReproducibilityExtended:
    """Test that random_state produces reproducible results with extended types."""

    def test_continuous_reproducibility(self):
        """Same random_state should produce same results with continuous dims."""
        opt1 = HillClimbingOptimizer(CONTINUOUS_SEARCH_SPACE, random_state=42)
        opt1.search(simple_objective, n_iter=20, verbosity=[])

        opt2 = HillClimbingOptimizer(CONTINUOUS_SEARCH_SPACE, random_state=42)
        opt2.search(simple_objective, n_iter=20, verbosity=[])

        assert opt1.best_score == opt2.best_score
        assert opt1.best_para == opt2.best_para

    def test_categorical_reproducibility(self):
        """Same random_state should produce same results with categorical dims."""
        opt1 = HillClimbingOptimizer(CATEGORICAL_SEARCH_SPACE, random_state=42)
        opt1.search(simple_objective, n_iter=20, verbosity=[])

        opt2 = HillClimbingOptimizer(CATEGORICAL_SEARCH_SPACE, random_state=42)
        opt2.search(simple_objective, n_iter=20, verbosity=[])

        assert opt1.best_score == opt2.best_score
        assert opt1.best_para == opt2.best_para

    def test_mixed_reproducibility(self):
        """Same random_state should produce same results with mixed dims."""
        opt1 = HillClimbingOptimizer(MIXED_SEARCH_SPACE, random_state=42)
        opt1.search(simple_objective, n_iter=20, verbosity=[])

        opt2 = HillClimbingOptimizer(MIXED_SEARCH_SPACE, random_state=42)
        opt2.search(simple_objective, n_iter=20, verbosity=[])

        assert opt1.best_score == opt2.best_score
        assert opt1.best_para == opt2.best_para


# =============================================================================
# Constraint tests with extended types
# =============================================================================


class TestConstraintsExtended:
    """Test constraints work with extended search space types."""

    def test_constraint_on_continuous(self):
        """Constraint should work on continuous dimension."""

        def constraint(params):
            return params["x"] > 0

        opt = RandomSearchOptimizer(
            CONTINUOUS_SEARCH_SPACE, constraints=[constraint], random_state=42
        )
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["x"] > 0

    def test_constraint_on_categorical(self):
        """Constraint should work on categorical dimension."""

        def constraint(params):
            return params["optimizer"] != "rmsprop"

        opt = RandomSearchOptimizer(
            CATEGORICAL_SEARCH_SPACE, constraints=[constraint], random_state=42
        )
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd"]

    def test_constraint_on_mixed(self):
        """Constraint should work on mixed dimension types."""

        def constraint(params):
            return params["x"] + params["y"] > 0

        opt = HillClimbingOptimizer(
            MIXED_SEARCH_SPACE, constraints=[constraint], random_state=42
        )
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["x"] + opt.best_para["y"] > 0

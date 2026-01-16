"""Integration tests for extended search space support.

Tests cover:
- Continuous dimensions: optimizer accepts tuples, returns float values
- Categorical dimensions: optimizer accepts lists, returns correct categories
- Mixed dimensions: all three types in one search space
- Constraint handling with mixed types
- Initialization strategies with mixed types
"""

import numpy as np
import pytest

from gradient_free_optimizers import (
    BayesianOptimizer,
    ForestOptimizer,
    GeneticAlgorithmOptimizer,
    HillClimbingOptimizer,
    ParticleSwarmOptimizer,
    RandomSearchOptimizer,
    SimulatedAnnealingOptimizer,
)

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def continuous_search_space():
    """Search space with only continuous dimensions."""
    return {
        "x": (-5.0, 5.0),
        "y": (0.0, 10.0),
    }


@pytest.fixture
def categorical_search_space():
    """Search space with only categorical dimensions."""
    return {
        "optimizer": ["adam", "sgd", "rmsprop"],
        "use_bias": [True, False],
    }


@pytest.fixture
def mixed_search_space():
    """Search space with all three dimension types."""
    return {
        "x": np.arange(-5, 5, 1),  # Discrete numerical
        "y": (-5.0, 5.0),  # Continuous
        "algorithm": ["adam", "sgd", "rmsprop"],  # Categorical
    }


def simple_objective(params):
    """Simple objective function for testing."""
    score = 0
    for key, value in params.items():
        if isinstance(value, int | float):
            score -= value**2
        elif isinstance(value, str):
            # Add small bonus for "adam"
            if value == "adam":
                score += 0.5
        elif isinstance(value, bool):
            if value:
                score += 0.1
    return score


# =============================================================================
# Continuous dimension tests
# =============================================================================


class TestContinuousDimensions:
    """Tests for continuous dimensions (tuples)."""

    def test_hill_climbing_continuous(self, continuous_search_space):
        """HillClimbingOptimizer should work with continuous dimensions."""
        opt = HillClimbingOptimizer(continuous_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert "x" in opt.best_para
        assert "y" in opt.best_para
        # Continuous values should be floats within bounds
        assert isinstance(opt.best_para["x"], float)
        assert isinstance(opt.best_para["y"], float)
        assert -5.0 <= opt.best_para["x"] <= 5.0
        assert 0.0 <= opt.best_para["y"] <= 10.0

    def test_random_search_continuous(self, continuous_search_space):
        """RandomSearchOptimizer should work with continuous dimensions."""
        opt = RandomSearchOptimizer(continuous_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert isinstance(opt.best_para["x"], float)
        assert isinstance(opt.best_para["y"], float)

    def test_simulated_annealing_continuous(self, continuous_search_space):
        """SimulatedAnnealingOptimizer should work with continuous dimensions."""
        opt = SimulatedAnnealingOptimizer(continuous_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["x"] <= 5.0
        assert 0.0 <= opt.best_para["y"] <= 10.0

    def test_particle_swarm_continuous(self, continuous_search_space):
        """ParticleSwarmOptimizer should work with continuous dimensions."""
        opt = ParticleSwarmOptimizer(
            continuous_search_space, population=5, random_state=42
        )
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["x"] <= 5.0

    def test_genetic_algorithm_continuous(self, continuous_search_space):
        """GeneticAlgorithmOptimizer should work with continuous dimensions."""
        opt = GeneticAlgorithmOptimizer(
            continuous_search_space, population=5, random_state=42
        )
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert isinstance(opt.best_para["x"], float)

    def test_forest_optimizer_continuous(self, continuous_search_space):
        """ForestOptimizer should work with continuous dimensions."""
        opt = ForestOptimizer(continuous_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["x"] <= 5.0


# =============================================================================
# Categorical dimension tests
# =============================================================================


class TestCategoricalDimensions:
    """Tests for categorical dimensions (lists)."""

    def test_hill_climbing_categorical(self, categorical_search_space):
        """HillClimbingOptimizer should work with categorical dimensions."""
        opt = HillClimbingOptimizer(categorical_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]
        assert opt.best_para["use_bias"] in [True, False]

    def test_random_search_categorical(self, categorical_search_space):
        """RandomSearchOptimizer should work with categorical dimensions."""
        opt = RandomSearchOptimizer(categorical_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]
        assert opt.best_para["use_bias"] in [True, False]

    def test_simulated_annealing_categorical(self, categorical_search_space):
        """SimulatedAnnealingOptimizer should work with categorical dimensions."""
        opt = SimulatedAnnealingOptimizer(categorical_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]

    def test_particle_swarm_categorical(self, categorical_search_space):
        """ParticleSwarmOptimizer should work with categorical dimensions."""
        opt = ParticleSwarmOptimizer(
            categorical_search_space, population=5, random_state=42
        )
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]

    def test_genetic_algorithm_categorical(self, categorical_search_space):
        """GeneticAlgorithmOptimizer should work with categorical dimensions."""
        opt = GeneticAlgorithmOptimizer(
            categorical_search_space, population=5, random_state=42
        )
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]

    def test_forest_optimizer_categorical(self, categorical_search_space):
        """ForestOptimizer should work with categorical dimensions."""
        opt = ForestOptimizer(categorical_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]


# =============================================================================
# Mixed dimension tests
# =============================================================================


class TestMixedDimensions:
    """Tests for mixed dimension types in the same search space."""

    def test_hill_climbing_mixed(self, mixed_search_space):
        """HillClimbingOptimizer should work with mixed dimensions."""
        opt = HillClimbingOptimizer(mixed_search_space, random_state=42)
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        # Discrete numerical (from np.arange)
        assert opt.best_para["x"] in list(range(-5, 5))
        # Continuous
        assert isinstance(opt.best_para["y"], float)
        assert -5.0 <= opt.best_para["y"] <= 5.0
        # Categorical
        assert opt.best_para["algorithm"] in ["adam", "sgd", "rmsprop"]

    def test_random_search_mixed(self, mixed_search_space):
        """RandomSearchOptimizer should work with mixed dimensions."""
        opt = RandomSearchOptimizer(mixed_search_space, random_state=42)
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["algorithm"] in ["adam", "sgd", "rmsprop"]

    def test_simulated_annealing_mixed(self, mixed_search_space):
        """SimulatedAnnealingOptimizer should work with mixed dimensions."""
        opt = SimulatedAnnealingOptimizer(mixed_search_space, random_state=42)
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["y"] <= 5.0

    def test_particle_swarm_mixed(self, mixed_search_space):
        """ParticleSwarmOptimizer should work with mixed dimensions."""
        opt = ParticleSwarmOptimizer(mixed_search_space, population=5, random_state=42)
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["algorithm"] in ["adam", "sgd", "rmsprop"]

    def test_genetic_algorithm_mixed(self, mixed_search_space):
        """GeneticAlgorithmOptimizer should work with mixed dimensions."""
        opt = GeneticAlgorithmOptimizer(
            mixed_search_space, population=5, random_state=42
        )
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["algorithm"] in ["adam", "sgd", "rmsprop"]

    def test_forest_optimizer_mixed(self, mixed_search_space):
        """ForestOptimizer should work with mixed dimensions."""
        opt = ForestOptimizer(mixed_search_space, random_state=42)
        opt.search(simple_objective, n_iter=30, verbosity=[])

        assert opt.best_para is not None
        assert -5.0 <= opt.best_para["y"] <= 5.0


# =============================================================================
# Constraint handling with mixed types
# =============================================================================


class TestConstraintsWithMixedTypes:
    """Tests for constraint handling with mixed dimension types."""

    def test_constraint_on_continuous_dimension(self):
        """Constraints should work on continuous dimensions."""
        search_space = {
            "x": (-10.0, 10.0),
            "y": (-10.0, 10.0),
        }

        # Constraint: x must be positive
        def constraint(params):
            return params["x"] > 0

        opt = RandomSearchOptimizer(
            search_space, constraints=[constraint], random_state=42
        )
        opt.search(simple_objective, n_iter=50, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["x"] > 0

    def test_constraint_on_categorical_dimension(self):
        """Constraints should work on categorical dimensions."""
        search_space = {
            "optimizer": ["adam", "sgd", "rmsprop", "adagrad"],
            "lr": (0.001, 0.1),
        }

        # Constraint: only allow "adam" or "sgd"
        def constraint(params):
            return params["optimizer"] in ["adam", "sgd"]

        opt = RandomSearchOptimizer(
            search_space, constraints=[constraint], random_state=42
        )
        opt.search(simple_objective, n_iter=50, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["optimizer"] in ["adam", "sgd"]

    def test_constraint_on_mixed_dimensions(self):
        """Constraints should work across mixed dimension types."""
        search_space = {
            "x": np.arange(-5, 5, 1),  # Discrete
            "y": (-5.0, 5.0),  # Continuous
            "algo": ["adam", "sgd"],  # Categorical
        }

        # Constraint: x + y must be positive
        def constraint(params):
            return params["x"] + params["y"] > 0

        opt = HillClimbingOptimizer(
            search_space, constraints=[constraint], random_state=42
        )
        opt.search(simple_objective, n_iter=50, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["x"] + opt.best_para["y"] > 0


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in extended search spaces."""

    def test_single_continuous_dimension(self):
        """Optimizer should work with a single continuous dimension."""
        search_space = {"x": (0.0, 1.0)}
        opt = HillClimbingOptimizer(search_space, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert 0.0 <= opt.best_para["x"] <= 1.0

    def test_single_categorical_dimension(self):
        """Optimizer should work with a single categorical dimension."""
        search_space = {"choice": ["a", "b", "c"]}
        opt = HillClimbingOptimizer(search_space, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["choice"] in ["a", "b", "c"]

    def test_categorical_with_none_values(self):
        """Categorical dimension with None values should work."""
        search_space = {"option": [None, "enabled", "disabled"]}
        opt = RandomSearchOptimizer(search_space, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["option"] in [None, "enabled", "disabled"]

    def test_categorical_with_numeric_values(self):
        """Categorical dimension with numeric values should work."""
        search_space = {"batch_size": [16, 32, 64, 128]}
        opt = HillClimbingOptimizer(search_space, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["batch_size"] in [16, 32, 64, 128]

    @pytest.mark.xfail(reason="Zero-range continuous dimension causes division by zero")
    def test_continuous_with_same_bounds(self):
        """Continuous dimension with equal min/max should work."""
        search_space = {"fixed": (5.0, 5.0)}
        opt = HillClimbingOptimizer(search_space, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["fixed"] == 5.0

    def test_two_element_list_is_categorical(self):
        """Two-element list should be categorical, not continuous."""
        search_space = {"flag": [True, False]}
        opt = HillClimbingOptimizer(search_space, random_state=42)
        opt.search(simple_objective, n_iter=10, verbosity=[])

        assert opt.best_para is not None
        assert opt.best_para["flag"] in [True, False]


# =============================================================================
# Search data and reproducibility
# =============================================================================


class TestSearchDataWithMixedTypes:
    """Tests for search_data output with mixed dimension types."""

    def test_search_data_contains_all_params(self, mixed_search_space):
        """search_data should contain all parameters with correct types."""
        opt = HillClimbingOptimizer(mixed_search_space, random_state=42)
        opt.search(simple_objective, n_iter=20, verbosity=[])

        data = opt.search_data
        assert "x" in data.columns
        assert "y" in data.columns
        assert "algorithm" in data.columns
        assert "score" in data.columns

    def test_reproducibility_with_random_state(self, mixed_search_space):
        """Same random_state should produce same results with mixed types."""
        opt1 = HillClimbingOptimizer(mixed_search_space, random_state=42)
        opt1.search(simple_objective, n_iter=20, verbosity=[])

        opt2 = HillClimbingOptimizer(mixed_search_space, random_state=42)
        opt2.search(simple_objective, n_iter=20, verbosity=[])

        assert opt1.best_score == opt2.best_score
        assert opt1.best_para == opt2.best_para

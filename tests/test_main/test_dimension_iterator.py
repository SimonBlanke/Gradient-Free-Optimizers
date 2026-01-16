"""Tests for the DimensionIteratorMixin.

Tests cover:
- iterate_position_typed() orchestration
- Sequential vs vectorized selection based on threshold
- _iterate_discrete_numerical_all() behavior
- _iterate_continuous_all() behavior
- _iterate_categorical_all() behavior
- VECTORIZATION_THRESHOLD behavior
"""

import numpy as np
import pytest

from gradient_free_optimizers import HillClimbingOptimizer
from gradient_free_optimizers._dimension_iterator import (
    VECTORIZATION_THRESHOLD,
    DimensionIteratorMixin,
)

# =============================================================================
# VECTORIZATION_THRESHOLD tests
# =============================================================================


class TestVectorizationThreshold:
    """Tests for the VECTORIZATION_THRESHOLD constant."""

    def test_threshold_is_positive_integer(self):
        """VECTORIZATION_THRESHOLD should be a positive integer."""
        assert isinstance(VECTORIZATION_THRESHOLD, int)
        assert VECTORIZATION_THRESHOLD > 0

    def test_threshold_default_value(self):
        """VECTORIZATION_THRESHOLD should be 1000 by default."""
        assert VECTORIZATION_THRESHOLD == 1000


# =============================================================================
# _can_vectorize() tests
# =============================================================================


class TestCanVectorize:
    """Tests for the _can_vectorize method."""

    def test_can_vectorize_below_threshold(self):
        """Should return False for small search spaces."""
        search_space = {f"x{i}": np.linspace(-5, 5, 10) for i in range(10)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        assert opt._can_vectorize() is False

    def test_can_vectorize_at_threshold(self):
        """Should return True at exactly VECTORIZATION_THRESHOLD dimensions."""
        search_space = {
            f"x{i}": np.linspace(-5, 5, 10) for i in range(VECTORIZATION_THRESHOLD)
        }
        opt = HillClimbingOptimizer(search_space, random_state=42)

        assert opt._can_vectorize() is True

    def test_can_vectorize_above_threshold(self):
        """Should return True for large search spaces."""
        search_space = {
            f"x{i}": np.linspace(-5, 5, 10)
            for i in range(VECTORIZATION_THRESHOLD + 100)
        }
        opt = HillClimbingOptimizer(search_space, random_state=42)

        assert opt._can_vectorize() is True


# =============================================================================
# DimensionIteratorMixin interface tests
# =============================================================================


class TestDimensionIteratorMixinInterface:
    """Tests for the DimensionIteratorMixin interface."""

    def test_mixin_provides_iterate_position_typed(self):
        """HillClimbingOptimizer should have iterate_position_typed method."""
        search_space = {"x": np.linspace(-5, 5, 100)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        assert hasattr(opt, "iterate_position_typed")
        assert callable(opt.iterate_position_typed)

    def test_mixin_provides_can_vectorize(self):
        """HillClimbingOptimizer should have _can_vectorize method."""
        search_space = {"x": np.linspace(-5, 5, 100)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        assert hasattr(opt, "_can_vectorize")
        assert callable(opt._can_vectorize)

    def test_iterate_returns_position_array(self):
        """iterate_position_typed should return a position array."""
        search_space = {"x": np.linspace(-5, 5, 100), "y": np.linspace(-5, 5, 100)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        # Initialize the optimizer
        def objective(p):
            return -(p["x"] ** 2 + p["y"] ** 2)

        opt.search(objective, n_iter=5, verbosity=[])

        # Now call iterate_position_typed
        current_pos = opt.pos_current
        new_pos = opt.iterate_position_typed(current_pos)

        assert new_pos is not None
        assert len(new_pos) == 2  # Two dimensions


# =============================================================================
# _iterate_discrete_numerical_all() tests
# =============================================================================


class TestIterateDiscreteNumericalAll:
    """Tests for the _iterate_discrete_numerical_all method."""

    def test_discrete_numerical_all_returns_list(self):
        """_iterate_discrete_numerical_all should return a list."""
        search_space = {"x": np.linspace(-5, 5, 100)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [50.0]
        max_positions = [99]

        result = opt._iterate_discrete_numerical_all(current_values, max_positions)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_discrete_numerical_all_applies_noise(self):
        """_iterate_discrete_numerical_all should apply Gaussian noise."""
        search_space = {"x": np.linspace(-5, 5, 100)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [50.0]
        max_positions = [99]

        # Call multiple times to verify noise is applied
        results = [
            opt._iterate_discrete_numerical_all(
                current_values, max_positions, epsilon=0.1
            )
            for _ in range(10)
        ]

        # Not all results should be identical (noise applied)
        unique_results = {tuple(r) for r in results}
        assert len(unique_results) > 1

    def test_discrete_numerical_all_respects_epsilon(self):
        """Larger epsilon should produce larger variations."""
        search_space = {"x": np.linspace(-5, 5, 100)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [50.0]
        max_positions = [99]

        # Small epsilon
        small_results = [
            opt._iterate_discrete_numerical_all(
                current_values, max_positions, epsilon=0.001
            )[0]
            for _ in range(20)
        ]
        small_variance = np.var(small_results)

        # Large epsilon
        large_results = [
            opt._iterate_discrete_numerical_all(
                current_values, max_positions, epsilon=0.5
            )[0]
            for _ in range(20)
        ]
        large_variance = np.var(large_results)

        # Larger epsilon should produce larger variance
        assert large_variance > small_variance


# =============================================================================
# _iterate_continuous_all() tests
# =============================================================================


class TestIterateContinuousAll:
    """Tests for the _iterate_continuous_all method."""

    def test_continuous_all_returns_list(self):
        """_iterate_continuous_all should return a list."""
        search_space = {"x": (-5.0, 5.0)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [0.0]
        bounds = [(-5.0, 5.0)]

        result = opt._iterate_continuous_all(current_values, bounds)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_continuous_all_applies_noise(self):
        """_iterate_continuous_all should apply Gaussian noise."""
        search_space = {"x": (-5.0, 5.0)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [0.0]
        bounds = [(-5.0, 5.0)]

        # Call multiple times to verify noise is applied
        results = [
            opt._iterate_continuous_all(current_values, bounds, epsilon=0.1)
            for _ in range(10)
        ]

        # Not all results should be identical
        unique_results = {tuple(r) for r in results}
        assert len(unique_results) > 1

    def test_continuous_all_scales_by_range(self):
        """Noise should scale with the range of the dimension."""
        search_space = {"x": (-5.0, 5.0)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        # Small range
        small_results = [
            opt._iterate_continuous_all([0.0], [(-0.1, 0.1)], epsilon=0.1)[0]
            for _ in range(20)
        ]
        small_variance = np.var(small_results)

        # Large range
        large_results = [
            opt._iterate_continuous_all([0.0], [(-100.0, 100.0)], epsilon=0.1)[0]
            for _ in range(20)
        ]
        large_variance = np.var(large_results)

        # Larger range should produce larger variance
        assert large_variance > small_variance


# =============================================================================
# _iterate_categorical_all() tests
# =============================================================================


class TestIterateCategoricalAll:
    """Tests for the _iterate_categorical_all method."""

    def test_categorical_all_returns_list(self):
        """_iterate_categorical_all should return a list."""
        search_space = {"algo": ["adam", "sgd", "rmsprop"]}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [0]  # Index into categorical list
        n_categories = [3]

        result = opt._iterate_categorical_all(current_values, n_categories)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_categorical_all_returns_valid_indices(self):
        """_iterate_categorical_all should return valid category indices."""
        search_space = {"algo": ["adam", "sgd", "rmsprop"]}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [0]
        n_categories = [3]

        for _ in range(20):
            result = opt._iterate_categorical_all(
                current_values, n_categories, epsilon=0.5
            )
            assert 0 <= result[0] < 3

    def test_categorical_all_sometimes_switches(self):
        """_iterate_categorical_all should sometimes switch categories."""
        search_space = {"algo": ["adam", "sgd", "rmsprop"]}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [0]
        n_categories = [3]

        # With high epsilon, should switch categories sometimes
        results = [
            opt._iterate_categorical_all(current_values, n_categories, epsilon=0.9)[0]
            for _ in range(50)
        ]

        # Should have switched at least once
        unique_values = set(results)
        assert len(unique_values) > 1

    def test_categorical_all_low_epsilon_keeps_value(self):
        """Very low epsilon should rarely switch categories."""
        search_space = {"algo": ["adam", "sgd", "rmsprop"]}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        current_values = [1]  # Start with sgd
        n_categories = [3]

        # With very low epsilon, should almost never switch
        results = [
            opt._iterate_categorical_all(current_values, n_categories, epsilon=0.001)[0]
            for _ in range(20)
        ]

        # Most results should be the original value
        original_count = results.count(1)
        assert original_count >= 18  # At least 90% should stay the same


# =============================================================================
# Integration tests
# =============================================================================


class TestIteratorIntegration:
    """Integration tests for the dimension iterator."""

    def test_sequential_mode_with_small_space(self):
        """Small search spaces should use sequential mode."""
        search_space = {"x": np.linspace(-5, 5, 100), "y": np.linspace(-5, 5, 100)}
        opt = HillClimbingOptimizer(search_space, random_state=42)

        def objective(p):
            return -(p["x"] ** 2 + p["y"] ** 2)

        # Should work without errors
        opt.search(objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert opt._can_vectorize() is False

    def test_mixed_types_with_sequential_mode(self):
        """Mixed dimension types should work in sequential mode."""
        search_space = {
            "x": np.arange(-5, 5, 1),  # Discrete
            "y": (-5.0, 5.0),  # Continuous
            "algo": ["adam", "sgd"],  # Categorical
        }
        opt = HillClimbingOptimizer(search_space, random_state=42)

        def objective(p):
            return -(p["x"] ** 2 + p["y"] ** 2)

        opt.search(objective, n_iter=20, verbosity=[])

        assert opt.best_para is not None
        assert "x" in opt.best_para
        assert "y" in opt.best_para
        assert "algo" in opt.best_para

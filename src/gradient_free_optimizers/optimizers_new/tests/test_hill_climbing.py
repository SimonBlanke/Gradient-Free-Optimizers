# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Tests for the new HillClimbingOptimizer implementation.
"""

import numpy as np
import pytest

from ..local_opt import HillClimbingOptimizer


class TestHillClimbingTemplatePattern:
    """Test the Template Method Pattern implementation for HillClimbing."""

    @pytest.fixture
    def mixed_search_space(self):
        """Create a search space with all three dimension types."""
        return {
            # Continuous dimensions (tuples)
            "learning_rate": (0.0001, 0.1),
            "momentum": (0.0, 1.0),
            # Categorical dimensions (lists)
            "activation": ["relu", "tanh", "sigmoid"],
            "optimizer": ["adam", "sgd", "rmsprop", "adagrad"],
            # Discrete dimensions (numpy arrays)
            "hidden_layers": np.array([1, 2, 3, 4, 5]),
            "batch_size": np.array([16, 32, 64, 128, 256]),
        }

    @pytest.fixture
    def optimizer(self, mixed_search_space):
        """Create an optimizer instance.

        Note: _setup_dimension_masks() is now auto-called in __init__
        """
        return HillClimbingOptimizer(
            search_space=mixed_search_space,
            random_state=42,
            epsilon=0.1,
            n_neighbours=3,
        )

    def test_dimension_mask_setup(self, optimizer):
        """Test that dimension masks are correctly initialized."""
        # Should have 2 continuous, 2 categorical, 2 discrete
        assert optimizer._continuous_mask.sum() == 2
        assert optimizer._categorical_mask.sum() == 2
        assert optimizer._discrete_mask.sum() == 2

        # Total should equal n_dims
        total = (
            optimizer._continuous_mask.sum()
            + optimizer._categorical_mask.sum()
            + optimizer._discrete_mask.sum()
        )
        assert total == 6

    def test_continuous_bounds_shape(self, optimizer):
        """Test continuous bounds array shape."""
        assert optimizer._continuous_bounds.shape == (2, 2)
        # Check bounds are reasonable
        assert np.all(
            optimizer._continuous_bounds[:, 0] < optimizer._continuous_bounds[:, 1]
        )

    def test_categorical_sizes_shape(self, optimizer):
        """Test categorical sizes array shape."""
        assert optimizer._categorical_sizes.shape == (2,)
        # Check sizes match search space
        assert optimizer._categorical_sizes[0] == 3  # activation
        assert optimizer._categorical_sizes[1] == 4  # optimizer

    def test_discrete_bounds_shape(self, optimizer):
        """Test discrete bounds array shape."""
        assert optimizer._discrete_bounds.shape == (2, 2)
        # Check bounds start at 0
        assert np.all(optimizer._discrete_bounds[:, 0] == 0)
        # Check max indices
        assert (
            optimizer._discrete_bounds[0, 1] == 4
        )  # hidden_layers (5 values -> max idx 4)
        assert (
            optimizer._discrete_bounds[1, 1] == 4
        )  # batch_size (5 values -> max idx 4)

    def test_iterate_continuous_batch(self, optimizer):
        """Test continuous batch iteration generates valid values."""
        # Set up state for parameterless method access
        current = np.array([0.05, 0.5])  # learning_rate, momentum
        full_pos = np.array([0.05, 0.5, 1, 2, 2.0, 3.0])
        optimizer._pos_current = full_pos

        new_vals = optimizer._iterate_continuous_batch()

        # Should return same shape as continuous mask
        assert new_vals.shape == current.shape
        # Values should be different (with very high probability given epsilon=0.1)
        assert not np.allclose(new_vals, current)

    def test_iterate_categorical_batch(self, optimizer):
        """Test categorical batch iteration generates valid indices."""
        # Set up state for parameterless method access
        full_pos = np.array([0.05, 0.5, 1, 2, 2.0, 3.0])
        optimizer._pos_current = full_pos
        current = np.array([1, 2])  # relu idx, rmsprop idx

        # Run many times to test probabilistic switching
        np.random.seed(42)
        all_same = True
        for _ in range(100):
            new_vals = optimizer._iterate_categorical_batch()
            if not np.array_equal(new_vals, current):
                all_same = False
                break

        # With epsilon=0.1, should eventually switch
        assert not all_same, "Categorical should switch with epsilon=0.1"

    def test_iterate_discrete_batch(self, optimizer):
        """Test discrete batch iteration generates values."""
        # Set up state for parameterless method access
        full_pos = np.array([0.05, 0.5, 1, 2, 2.0, 3.0])
        optimizer._pos_current = full_pos
        current = np.array([2.0, 3.0])  # hidden_layers idx, batch_size idx

        new_vals = optimizer._iterate_discrete_batch()

        # Should return same shape as discrete mask
        assert new_vals.shape == current.shape
        # Values should be different (with high probability)
        assert not np.allclose(new_vals, current)

    def test_clip_position_continuous(self, optimizer):
        """Test clipping for continuous dimensions."""
        # Create a position with out-of-bounds continuous values
        position = np.zeros(6)
        # Set continuous values (first 2 dimensions based on mask)
        cont_indices = np.where(optimizer._continuous_mask)[0]
        position[cont_indices[0]] = -1.0  # below min
        position[cont_indices[1]] = 2.0  # above max

        clipped = optimizer._clip_position(position)

        # Should be clipped to bounds
        assert clipped[cont_indices[0]] >= optimizer._continuous_bounds[0, 0]
        assert clipped[cont_indices[1]] <= optimizer._continuous_bounds[1, 1]

    def test_clip_position_categorical(self, optimizer):
        """Test clipping for categorical dimensions."""
        position = np.zeros(6)
        cat_indices = np.where(optimizer._categorical_mask)[0]
        position[cat_indices[0]] = -1.0  # below 0
        position[cat_indices[1]] = 10.0  # above max category

        clipped = optimizer._clip_position(position)

        # Should be clipped to [0, n_categories-1]
        assert clipped[cat_indices[0]] == 0
        assert clipped[cat_indices[1]] == optimizer._categorical_sizes[1] - 1

    def test_clip_position_discrete(self, optimizer):
        """Test clipping for discrete dimensions."""
        position = np.zeros(6)
        disc_indices = np.where(optimizer._discrete_mask)[0]
        position[disc_indices[0]] = -2.0  # below 0
        position[disc_indices[1]] = 100.0  # above max

        clipped = optimizer._clip_position(position)

        # Should be clipped to valid range
        assert clipped[disc_indices[0]] == 0
        assert clipped[disc_indices[1]] == optimizer._discrete_bounds[1, 1]

    def test_full_iterate_cycle(self, optimizer):
        """Test a complete iterate cycle with clipping."""
        # Set up initial position
        optimizer.pos_current = np.array([0.05, 0.5, 1, 2, 2.0, 3.0])

        # Run iterate
        new_pos = optimizer.iterate()

        # Should return valid position
        assert new_pos.shape == optimizer.pos_current.shape

        # Continuous should be within bounds
        cont_idx = np.where(optimizer._continuous_mask)[0]
        for i, idx in enumerate(cont_idx):
            assert (
                optimizer._continuous_bounds[i, 0]
                <= new_pos[idx]
                <= optimizer._continuous_bounds[i, 1]
            )

        # Categorical should be integer indices
        cat_idx = np.where(optimizer._categorical_mask)[0]
        for i, idx in enumerate(cat_idx):
            assert 0 <= new_pos[idx] < optimizer._categorical_sizes[i]
            assert new_pos[idx] == int(new_pos[idx])

        # Discrete should be integer indices
        disc_idx = np.where(optimizer._discrete_mask)[0]
        for i, idx in enumerate(disc_idx):
            assert (
                optimizer._discrete_bounds[i, 0]
                <= new_pos[idx]
                <= optimizer._discrete_bounds[i, 1]
            )
            assert new_pos[idx] == int(new_pos[idx])


class TestHillClimbingDistributions:
    """Test different noise distributions."""

    @pytest.fixture
    def continuous_space(self):
        """Simple continuous-only search space."""
        return {
            "x": (0.0, 10.0),
            "y": (0.0, 10.0),
        }

    @pytest.mark.parametrize("distribution", ["normal", "laplace", "logistic"])
    def test_distribution_types(self, continuous_space, distribution):
        """Test that all distribution types work."""
        opt = HillClimbingOptimizer(
            search_space=continuous_space,
            random_state=42,
            distribution=distribution,
        )
        # _setup_dimension_masks() is now auto-called in __init__
        opt.pos_current = np.array([5.0, 5.0])

        new_pos = opt.iterate()

        assert new_pos.shape == (2,)
        assert 0.0 <= new_pos[0] <= 10.0
        assert 0.0 <= new_pos[1] <= 10.0

    def test_invalid_distribution_raises(self, continuous_space):
        """Test that invalid distribution raises ValueError."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            HillClimbingOptimizer(
                search_space=continuous_space,
                distribution="invalid",
            )


class TestHillClimbingEvaluate:
    """Test the evaluate method (Template Pattern)."""

    @pytest.fixture
    def optimizer(self):
        """Create an optimizer for evaluation testing."""
        search_space = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
        opt = HillClimbingOptimizer(
            search_space=search_space,
            random_state=42,
            n_neighbours=3,
        )
        # _setup_dimension_masks() is now auto-called in __init__
        opt.pos_current = np.array([5.0, 5.0])
        opt.pos_new = np.array([5.0, 5.0])
        return opt

    def test_evaluate_tracks_scores(self, optimizer):
        """Test that evaluate tracks scores and positions via _track_score."""
        optimizer.pos_new = np.array([6.0, 6.0])
        optimizer.evaluate(1.0)

        # CoreOptimizer.evaluate calls _track_score which updates these
        assert len(optimizer.scores_valid) == 1
        assert optimizer.scores_valid[0] == 1.0
        assert optimizer.nth_trial == 1

    def test_evaluate_n_neighbours_update(self, optimizer):
        """Test that _evaluate updates after n_neighbours trials."""
        positions = [
            np.array([6.0, 6.0]),
            np.array([7.0, 7.0]),
            np.array([8.0, 8.0]),
        ]
        scores = [1.0, 3.0, 2.0]  # Best is index 1

        for pos, score in zip(positions, scores):
            optimizer.pos_new = pos.copy()
            optimizer.evaluate(score)

        # After 3 evaluations (n_neighbours=3), should update to best
        assert optimizer.score_best == 3.0
        assert np.array_equal(optimizer.pos_best, positions[1])

    def test_template_pattern_separation(self, optimizer):
        """Test that tracking happens before algorithm-specific logic."""
        # First evaluation - tracking should happen immediately
        optimizer.pos_new = np.array([6.0, 6.0])
        optimizer.evaluate(1.0)

        # Verify tracking was done (by CoreOptimizer._track_score)
        assert optimizer.nth_trial == 1
        assert len(optimizer.scores_valid) == 1

        # After first evaluation, best IS set (CoreOptimizer initializes it)
        # This is needed for Search progress tracking
        assert optimizer.score_best == 1.0

        # Complete 2 more evaluations
        optimizer.pos_new = np.array([7.0, 7.0])
        optimizer.evaluate(2.0)
        optimizer.pos_new = np.array([8.0, 8.0])
        optimizer.evaluate(3.0)

        # After n_neighbours (3), _evaluate runs and updates best to the best score
        assert optimizer.score_best == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

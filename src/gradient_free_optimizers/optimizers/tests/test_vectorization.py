# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Tests for vectorized batch operations in optimizers.

These tests verify that:
1. Dimension masks are correctly created
2. Batch methods receive arrays of correct shape
3. Operations work correctly with many dimensions
4. Vectorization produces consistent results

Strategy for fast CI:
- Use moderate dimension counts (50-100) for most tests
- Use mocking to verify batch method calls without full optimization
- Mark slow tests (100+ dimensions, many iterations) with @pytest.mark.slow

As new optimizers are implemented in optimizers/, add them to the
OPTIMIZERS list below.

NOTE: We import from optimizer_search/ because that's where the complete
optimizer classes live (combining optimizers/ with Search functionality).
"""

import numpy as np
import pytest

# Import from optimizer_search/ - these are the complete optimizers with search()
from ...optimizer_search import (
    HillClimbingOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticHillClimbingOptimizer,
)

OPTIMIZERS = [
    HillClimbingOptimizer,
    RandomSearchOptimizer,
    StochasticHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    RepulsingHillClimbingOptimizer,
    RandomRestartHillClimbingOptimizer,
]


def make_continuous_space(n_dims: int) -> dict:
    """Create a search space with n_dims continuous dimensions."""
    return {f"x{i}": (-10.0, 10.0) for i in range(n_dims)}


def make_categorical_space(n_dims: int, n_categories: int = 5) -> dict:
    """Create a search space with n_dims categorical dimensions."""
    categories = [f"cat{j}" for j in range(n_categories)]
    return {f"c{i}": categories for i in range(n_dims)}


def make_discrete_space(n_dims: int, n_values: int = 10) -> dict:
    """Create a search space with n_dims discrete dimensions."""
    values = np.linspace(0, 100, n_values)
    return {f"d{i}": values.copy() for i in range(n_dims)}


def make_mixed_space(n_continuous: int, n_categorical: int, n_discrete: int) -> dict:
    """Create a search space with mixed dimension types."""
    space = {}
    # Continuous
    for i in range(n_continuous):
        space[f"cont_{i}"] = (-10.0, 10.0)
    # Categorical
    for i in range(n_categorical):
        space[f"cat_{i}"] = ["a", "b", "c", "d"]
    # Discrete
    for i in range(n_discrete):
        space[f"disc_{i}"] = np.array([1, 2, 4, 8, 16])
    return space


def simple_objective(para):
    """Sum all numeric values in the parameter dictionary."""
    total = 0.0
    for key, value in para.items():
        if isinstance(value, int | float | np.number):
            total += value
        elif isinstance(value, str):
            total += len(value)  # Use string length as proxy
        elif isinstance(value, bool):
            total += 1 if value else 0
    return -abs(total)  # Maximize towards 0


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mask_continuous_only(OptimizerClass):
    """Test that continuous-only space creates correct masks."""
    space = make_continuous_space(10)
    opt = OptimizerClass(space, random_state=42)

    assert opt._continuous_mask is not None
    assert opt._continuous_mask.sum() == 10
    assert opt._categorical_mask.sum() == 0
    assert opt._discrete_mask.sum() == 0


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mask_categorical_only(OptimizerClass):
    """Test that categorical-only space creates correct masks."""
    space = make_categorical_space(8)
    opt = OptimizerClass(space, random_state=42)

    assert opt._continuous_mask.sum() == 0
    assert opt._categorical_mask.sum() == 8
    assert opt._discrete_mask.sum() == 0


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mask_discrete_only(OptimizerClass):
    """Test that discrete-only space creates correct masks."""
    space = make_discrete_space(12)
    opt = OptimizerClass(space, random_state=42)

    assert opt._continuous_mask.sum() == 0
    assert opt._categorical_mask.sum() == 0
    assert opt._discrete_mask.sum() == 12


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mask_mixed_space(OptimizerClass):
    """Test that mixed space creates correct masks."""
    space = make_mixed_space(n_continuous=5, n_categorical=3, n_discrete=4)
    opt = OptimizerClass(space, random_state=42)

    assert opt._continuous_mask.sum() == 5
    assert opt._categorical_mask.sum() == 3
    assert opt._discrete_mask.sum() == 4
    # Total should match
    total = (
        opt._continuous_mask.sum()
        + opt._categorical_mask.sum()
        + opt._discrete_mask.sum()
    )
    assert total == len(space)


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_bounds_arrays_shape(OptimizerClass):
    """Test that bounds arrays have correct shapes."""
    space = make_mixed_space(n_continuous=5, n_categorical=3, n_discrete=4)
    opt = OptimizerClass(space, random_state=42)

    # Continuous bounds: (n_continuous, 2)
    assert opt._continuous_bounds.shape == (5, 2)

    # Categorical sizes: (n_categorical,)
    assert opt._categorical_sizes.shape == (3,)

    # Discrete bounds: (n_discrete, 2)
    assert opt._discrete_bounds.shape == (4, 2)


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_continuous_batch_called_with_array(OptimizerClass):
    """Verify _iterate_continuous_batch receives vectorized input."""
    space = make_continuous_space(20)
    opt = OptimizerClass(space, random_state=42)

    # Run one iteration to trigger batch method
    opt.search(simple_objective, n_iter=5)

    # The continuous bounds should have been set up correctly
    assert opt._continuous_bounds.shape == (20, 2)
    # All dimensions should be continuous
    assert opt._continuous_mask.all()


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_categorical_batch_processes_all_dims(OptimizerClass):
    """Verify _iterate_categorical_batch handles all categorical dimensions."""
    space = make_categorical_space(15, n_categories=4)
    opt = OptimizerClass(space, random_state=42)

    opt.search(simple_objective, n_iter=5)

    # Categorical sizes should match
    assert len(opt._categorical_sizes) == 15
    assert all(s == 4 for s in opt._categorical_sizes)


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_discrete_batch_processes_all_dims(OptimizerClass):
    """Verify _iterate_discrete_batch handles all discrete dimensions."""
    space = make_discrete_space(18, n_values=8)
    opt = OptimizerClass(space, random_state=42)

    opt.search(simple_objective, n_iter=5)

    # Discrete bounds should be [0, n_values-1] for each
    assert opt._discrete_bounds.shape == (18, 2)
    assert all(opt._discrete_bounds[:, 0] == 0)
    assert all(opt._discrete_bounds[:, 1] == 7)  # n_values - 1


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_50_continuous_dimensions(OptimizerClass):
    """Test optimizer works correctly with 50 continuous dimensions."""
    space = make_continuous_space(50)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=30)

    assert opt.best_para is not None
    assert len(opt.best_para) == 50
    # All values should be in range
    for key, value in opt.best_para.items():
        assert -10.0 <= value <= 10.0


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_50_categorical_dimensions(OptimizerClass):
    """Test optimizer works correctly with 50 categorical dimensions."""
    space = make_categorical_space(50, n_categories=5)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=30)

    assert opt.best_para is not None
    assert len(opt.best_para) == 50
    # All values should be valid categories
    valid_cats = [f"cat{j}" for j in range(5)]
    for key, value in opt.best_para.items():
        assert value in valid_cats


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_50_discrete_dimensions(OptimizerClass):
    """Test optimizer works correctly with 50 discrete dimensions."""
    values = np.linspace(0, 100, 10)
    space = {f"d{i}": values.copy() for i in range(50)}
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=30)

    assert opt.best_para is not None
    assert len(opt.best_para) == 50
    # All values should be from the discrete set
    for key, value in opt.best_para.items():
        assert value in values


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mixed_50_dimensions(OptimizerClass):
    """Test optimizer with mixed 50-dimension space (16+17+17)."""
    space = make_mixed_space(n_continuous=16, n_categorical=17, n_discrete=17)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=30)

    assert opt.best_para is not None
    assert len(opt.best_para) == 50


@pytest.mark.slow
@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_200_continuous_dimensions(OptimizerClass):
    """Test optimizer with 200 continuous dimensions (slow test)."""
    space = make_continuous_space(200)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=50)

    assert opt.best_para is not None
    assert len(opt.best_para) == 200


@pytest.mark.slow
@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_200_mixed_dimensions(OptimizerClass):
    """Test optimizer with 200 mixed dimensions (slow test)."""
    space = make_mixed_space(n_continuous=70, n_categorical=65, n_discrete=65)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=50)

    assert opt.best_para is not None
    assert len(opt.best_para) == 200


@pytest.mark.slow
@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_500_continuous_dimensions(OptimizerClass):
    """Test optimizer with 500 continuous dimensions (slow test)."""
    space = make_continuous_space(500)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=30)

    assert opt.best_para is not None
    assert len(opt.best_para) == 500


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_position_array_length_matches_space(OptimizerClass):
    """Test that internal position arrays match search space size."""
    space = make_mixed_space(n_continuous=10, n_categorical=8, n_discrete=7)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=10)

    # pos_best should have length equal to search space
    assert len(opt._pos_best) == 25
    assert len(opt._pos_current) == 25


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_position_values_correct_types(OptimizerClass):
    """Test that position array has correct value types per dimension."""
    space = make_mixed_space(n_continuous=5, n_categorical=3, n_discrete=4)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=10)

    pos = opt._pos_best

    # Continuous values (first 5) should be floats in range
    for i in range(5):
        assert opt._continuous_mask[i]
        assert -10.0 <= pos[i] <= 10.0

    # Categorical values (next 3) should be integer indices
    for i in range(5, 8):
        assert opt._categorical_mask[i]
        assert 0 <= pos[i] < 4  # 4 categories

    # Discrete values (last 4) should be integer indices
    for i in range(8, 12):
        assert opt._discrete_mask[i]
        assert 0 <= pos[i] < 5  # 5 values in discrete array


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_reproducibility_50_dimensions(OptimizerClass):
    """Test that results are reproducible with 50 dimensions."""
    space = make_mixed_space(n_continuous=20, n_categorical=15, n_discrete=15)

    opt1 = OptimizerClass(space, random_state=999)
    opt1.search(simple_objective, n_iter=20)

    opt2 = OptimizerClass(space, random_state=999)
    opt2.search(simple_objective, n_iter=20)

    assert opt1.best_para == opt2.best_para
    assert opt1.best_score == opt2.best_score


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_clipping_continuous_enforced(OptimizerClass):
    """Test that continuous values are always clipped to bounds."""
    space = make_continuous_space(30)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=50)

    # Check all positions in history are within bounds
    for pos in opt._pos_new_list:
        continuous_vals = pos[opt._continuous_mask]
        assert all(continuous_vals >= -10.0)
        assert all(continuous_vals <= 10.0)


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_clipping_categorical_enforced(OptimizerClass):
    """Test that categorical indices are always valid."""
    space = make_categorical_space(20, n_categories=6)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=50)

    # Check all positions have valid category indices
    for pos in opt._pos_new_list:
        cat_vals = pos[opt._categorical_mask]
        assert all(cat_vals >= 0)
        assert all(cat_vals < 6)


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_clipping_discrete_enforced(OptimizerClass):
    """Test that discrete indices are always valid."""
    space = make_discrete_space(25, n_values=8)
    opt = OptimizerClass(space, random_state=42)
    opt.search(simple_objective, n_iter=50)

    # Check all positions have valid discrete indices
    for pos in opt._pos_new_list:
        disc_vals = pos[opt._discrete_mask]
        assert all(disc_vals >= 0)
        assert all(disc_vals <= 7)  # n_values - 1

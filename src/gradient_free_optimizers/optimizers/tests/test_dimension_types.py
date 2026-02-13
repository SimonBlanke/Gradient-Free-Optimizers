# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Tests for dimension type support in optimizers.

These tests verify that each optimizer correctly handles:
- Continuous dimensions (tuple format): (min, max)
- Categorical dimensions (list format): ["a", "b", "c"]
- Discrete dimensions (np.ndarray format): np.array([1, 2, 4, 8])
- Mixed dimensions (all three types combined)

As new optimizers are implemented in optimizers/, add them to the
OPTIMIZERS list below. All optimizers should pass these tests.

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


@pytest.fixture
def continuous_only_space():
    """Search space with only continuous dimensions."""
    return {
        "x": (-10.0, 10.0),
        "y": (0.0, 1.0),
        "z": (-100.0, 100.0),
    }


@pytest.fixture
def categorical_only_space():
    """Search space with only categorical dimensions."""
    return {
        "activation": ["relu", "tanh", "sigmoid"],
        "optimizer": ["adam", "sgd", "rmsprop", "adagrad"],
        "use_dropout": [True, False],
    }


@pytest.fixture
def discrete_only_space():
    """Search space with only discrete numerical dimensions."""
    return {
        "n_layers": np.array([1, 2, 3, 4, 5]),
        "batch_size": np.array([16, 32, 64, 128, 256]),
        "learning_rate": np.array([0.0001, 0.001, 0.01, 0.1]),  # log-scale
    }


@pytest.fixture
def mixed_space():
    """Search space with all three dimension types."""
    return {
        "lr": (0.0001, 0.1),  # continuous
        "activation": ["relu", "tanh", "sigmoid"],  # categorical
        "n_layers": np.array([1, 2, 4, 8]),  # discrete
        "momentum": (0.0, 0.99),  # continuous
        "use_bn": [True, False],  # categorical
    }


def make_objective(search_space):
    """Create a simple objective function that works with any search space."""

    def objective(para):
        # Sum up all numeric values (for continuous/discrete)
        # For categorical, use index position as proxy
        score = 0.0
        for name, value in para.items():
            dim_def = search_space[name]
            if isinstance(dim_def, tuple):
                # Continuous: prefer middle of range
                mid = (dim_def[0] + dim_def[1]) / 2
                score -= abs(value - mid)
            elif isinstance(dim_def, list):
                # Categorical: prefer first option
                score -= dim_def.index(value)
            elif isinstance(dim_def, np.ndarray):
                # Discrete: prefer middle value
                mid_idx = len(dim_def) // 2
                actual_idx = np.where(dim_def == value)[0][0]
                score -= abs(actual_idx - mid_idx)
        return score

    return objective


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_continuous_only_runs(OptimizerClass, continuous_only_space):
    """Test that optimizer runs with only continuous dimensions."""
    opt = OptimizerClass(continuous_only_space, random_state=42)
    objective = make_objective(continuous_only_space)
    opt.search(objective, n_iter=20)

    assert opt.best_para is not None
    assert opt.best_score is not None


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_continuous_values_in_range(OptimizerClass, continuous_only_space):
    """Test that continuous results are within the specified bounds."""
    opt = OptimizerClass(continuous_only_space, random_state=42)
    objective = make_objective(continuous_only_space)
    opt.search(objective, n_iter=20)

    best = opt.best_para
    for name, (min_val, max_val) in continuous_only_space.items():
        assert (
            min_val <= best[name] <= max_val
        ), f"{name}: {best[name]} not in [{min_val}, {max_val}]"


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_continuous_types_are_float(OptimizerClass, continuous_only_space):
    """Test that continuous results are float types."""
    opt = OptimizerClass(continuous_only_space, random_state=42)
    objective = make_objective(continuous_only_space)
    opt.search(objective, n_iter=20)

    best = opt.best_para
    for name in continuous_only_space:
        assert isinstance(
            best[name], float | np.floating
        ), f"{name}: expected float, got {type(best[name])}"


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_categorical_only_runs(OptimizerClass, categorical_only_space):
    """Test that optimizer runs with only categorical dimensions."""
    opt = OptimizerClass(categorical_only_space, random_state=42)
    objective = make_objective(categorical_only_space)
    opt.search(objective, n_iter=20)

    assert opt.best_para is not None
    assert opt.best_score is not None


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_categorical_values_are_valid(OptimizerClass, categorical_only_space):
    """Test that categorical results are from the allowed options."""
    opt = OptimizerClass(categorical_only_space, random_state=42)
    objective = make_objective(categorical_only_space)
    opt.search(objective, n_iter=20)

    best = opt.best_para
    for name, options in categorical_only_space.items():
        assert best[name] in options, f"{name}: {best[name]} not in {options}"


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_categorical_preserves_types(OptimizerClass, categorical_only_space):
    """Test that categorical results preserve original types (str, bool, etc.)."""
    opt = OptimizerClass(categorical_only_space, random_state=42)
    objective = make_objective(categorical_only_space)
    opt.search(objective, n_iter=20)

    best = opt.best_para
    # "use_dropout" should be a bool, not a string
    assert isinstance(
        best["use_dropout"], bool | np.bool_
    ), f"use_dropout: expected bool, got {type(best['use_dropout'])}"
    # "activation" should be a string
    assert isinstance(
        best["activation"], str
    ), f"activation: expected str, got {type(best['activation'])}"


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_discrete_only_runs(OptimizerClass, discrete_only_space):
    """Test that optimizer runs with only discrete numerical dimensions."""
    opt = OptimizerClass(discrete_only_space, random_state=42)
    objective = make_objective(discrete_only_space)
    opt.search(objective, n_iter=20)

    assert opt.best_para is not None
    assert opt.best_score is not None


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_discrete_values_are_valid(OptimizerClass, discrete_only_space):
    """Test that discrete results are from the allowed values."""
    opt = OptimizerClass(discrete_only_space, random_state=42)
    objective = make_objective(discrete_only_space)
    opt.search(objective, n_iter=20)

    best = opt.best_para
    for name, values in discrete_only_space.items():
        assert best[name] in values, f"{name}: {best[name]} not in {list(values)}"


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_discrete_values_exact_match(OptimizerClass, discrete_only_space):
    """Test that discrete results are exact values from the array (no interpolation)."""
    opt = OptimizerClass(discrete_only_space, random_state=42)
    objective = make_objective(discrete_only_space)
    opt.search(objective, n_iter=20)

    best = opt.best_para
    for name, values in discrete_only_space.items():
        # Check exact match (not just approximate)
        matches = [best[name] == v for v in values]
        assert any(
            matches
        ), f"{name}: {best[name]} is not an exact match in {list(values)}"


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mixed_dimensions_runs(OptimizerClass, mixed_space):
    """Test that optimizer runs with mixed dimension types."""
    opt = OptimizerClass(mixed_space, random_state=42)
    objective = make_objective(mixed_space)
    opt.search(objective, n_iter=30)

    assert opt.best_para is not None
    assert opt.best_score is not None


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mixed_dimensions_correct_types(OptimizerClass, mixed_space):
    """Test that mixed search space returns correct types for each dimension."""
    opt = OptimizerClass(mixed_space, random_state=42)
    objective = make_objective(mixed_space)
    opt.search(objective, n_iter=30)

    best = opt.best_para

    # Continuous: should be float
    assert isinstance(best["lr"], float | np.floating)
    assert isinstance(best["momentum"], float | np.floating)

    # Categorical: should match original type
    assert best["activation"] in ["relu", "tanh", "sigmoid"]
    assert best["use_bn"] in [True, False]

    # Discrete: should be exact value from array
    assert best["n_layers"] in [1, 2, 4, 8]


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_mixed_dimensions_values_in_range(OptimizerClass, mixed_space):
    """Test that all values are within their respective valid ranges."""
    opt = OptimizerClass(mixed_space, random_state=42)
    objective = make_objective(mixed_space)
    opt.search(objective, n_iter=30)

    best = opt.best_para

    # Continuous bounds
    assert 0.0001 <= best["lr"] <= 0.1
    assert 0.0 <= best["momentum"] <= 0.99

    # Categorical membership
    assert best["activation"] in ["relu", "tanh", "sigmoid"]
    assert best["use_bn"] in [True, False]

    # Discrete membership
    assert best["n_layers"] in [1, 2, 4, 8]


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_single_continuous_dimension(OptimizerClass):
    """Test with a single continuous dimension."""
    space = {"x": (-5.0, 5.0)}
    opt = OptimizerClass(space, random_state=42)
    opt.search(lambda p: -(p["x"] ** 2), n_iter=20)

    assert opt.best_para is not None
    assert -5.0 <= opt.best_para["x"] <= 5.0


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_single_categorical_dimension(OptimizerClass):
    """Test with a single categorical dimension."""
    space = {"choice": ["a", "b", "c"]}
    opt = OptimizerClass(space, random_state=42)
    opt.search(lambda p: {"a": 1, "b": 2, "c": 0}[p["choice"]], n_iter=20)

    assert opt.best_para is not None
    assert opt.best_para["choice"] in ["a", "b", "c"]


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_single_discrete_dimension(OptimizerClass):
    """Test with a single discrete dimension."""
    space = {"n": np.array([1, 2, 3, 4, 5])}
    opt = OptimizerClass(space, random_state=42)
    opt.search(lambda p: -abs(p["n"] - 3), n_iter=20)

    assert opt.best_para is not None
    assert opt.best_para["n"] in [1, 2, 3, 4, 5]


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_two_category_dimension(OptimizerClass):
    """Test categorical dimension with only two options (binary)."""
    space = {"flag": [True, False]}
    opt = OptimizerClass(space, random_state=42)
    opt.search(lambda p: 1 if p["flag"] else 0, n_iter=10)

    assert opt.best_para is not None
    assert opt.best_para["flag"] in [True, False]


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_narrow_continuous_range(OptimizerClass):
    """Test continuous dimension with very narrow range."""
    space = {"x": (0.999, 1.001)}
    opt = OptimizerClass(space, random_state=42)
    opt.search(lambda p: -abs(p["x"] - 1.0), n_iter=20)

    assert opt.best_para is not None
    assert 0.999 <= opt.best_para["x"] <= 1.001


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_large_categorical_space(OptimizerClass):
    """Test categorical dimension with many options."""
    space = {"letter": list("abcdefghijklmnopqrstuvwxyz")}
    opt = OptimizerClass(space, random_state=42)
    opt.search(lambda p: ord(p["letter"]), n_iter=30)

    assert opt.best_para is not None
    assert opt.best_para["letter"] in list("abcdefghijklmnopqrstuvwxyz")


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_reproducibility_continuous(OptimizerClass, continuous_only_space):
    """Test that results are reproducible with same random_state (continuous)."""
    objective = make_objective(continuous_only_space)

    opt1 = OptimizerClass(continuous_only_space, random_state=123)
    opt1.search(objective, n_iter=15)

    opt2 = OptimizerClass(continuous_only_space, random_state=123)
    opt2.search(objective, n_iter=15)

    assert opt1.best_para == opt2.best_para
    assert opt1.best_score == opt2.best_score


@pytest.mark.parametrize("OptimizerClass", OPTIMIZERS)
def test_reproducibility_mixed(OptimizerClass, mixed_space):
    """Test that results are reproducible with same random_state (mixed)."""
    objective = make_objective(mixed_space)

    opt1 = OptimizerClass(mixed_space, random_state=456)
    opt1.search(objective, n_iter=20)

    opt2 = OptimizerClass(mixed_space, random_state=456)
    opt2.search(objective, n_iter=20)

    assert opt1.best_para == opt2.best_para
    assert opt1.best_score == opt2.best_score

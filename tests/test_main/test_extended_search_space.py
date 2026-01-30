"""Tests for extended search space support (categorical and continuous dimensions).

Core test philosophy:
- Use pytest.parametrize to loop over ALL optimizers
- Test actual BEHAVIOR: does the optimizer explore different values?
- Keep tests simple and focused
"""

import numpy as np
import pytest

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

# =============================================================================
# All optimizers to test
# =============================================================================

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

# Optimizers that require a population parameter
POPULATION_OPTIMIZERS = {
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer,
}


def get_optimizer_kwargs(opt_class):
    """Return extra kwargs needed for specific optimizers."""
    if opt_class in POPULATION_OPTIMIZERS:
        return {"population": 5}
    return {}


def optimizer_id(opt_class):
    """Generate readable test ID from optimizer class."""
    return opt_class.__name__


# =============================================================================
# Core exploration tests
# =============================================================================


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_categorical_exploration(optimizer_class):
    """Test that optimizer explores categorical dimensions.

    The key assertion: after N iterations, the optimizer should have
    tried more than one unique value in the categorical dimension.
    This verifies the algorithm actually moves in categorical space.
    """
    search_space = {
        "category": ["a", "b", "c", "d", "e"],
    }

    def objective(params):
        # Give different scores to encourage exploration
        scores = {"a": 0.1, "b": 0.2, "c": 0.5, "d": 0.3, "e": 0.4}
        return scores[params["category"]]

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=30, verbosity=[])

    # Core assertion: optimizer explored multiple values
    unique_values = opt.search_data["category"].nunique()
    assert unique_values > 1, (
        f"{optimizer_class.__name__} did not explore categorical dimension "
        f"(only found {unique_values} unique value(s))"
    )


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_continuous_exploration(optimizer_class):
    """Test that optimizer explores continuous dimensions.

    The key assertion: after N iterations, the optimizer should have
    tried more than one unique value in the continuous dimension.
    This verifies the algorithm actually moves in continuous space.
    """
    search_space = {
        "x": (0.0, 10.0),
    }

    def objective(params):
        # Quadratic with optimum at x=5
        return -((params["x"] - 5) ** 2)

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=30, verbosity=[])

    # Core assertion: optimizer explored multiple values
    unique_values = opt.search_data["x"].nunique()
    assert unique_values > 1, (
        f"{optimizer_class.__name__} did not explore continuous dimension "
        f"(only found {unique_values} unique value(s))"
    )


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_mixed_exploration(optimizer_class):
    """Test that optimizer explores mixed dimension types.

    Combines discrete, continuous, and categorical in one search space.
    """
    search_space = {
        "discrete": np.arange(0, 10),
        "continuous": (0.0, 10.0),
        "categorical": ["opt1", "opt2", "opt3"],
    }

    def objective(params):
        score = 0
        # Prefer middle values for numeric
        score -= (params["discrete"] - 5) ** 2
        score -= (params["continuous"] - 5) ** 2
        # Prefer opt2
        if params["categorical"] == "opt2":
            score += 1
        return score

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=30, verbosity=[])

    # All dimensions should be explored
    assert opt.search_data["discrete"].nunique() > 1
    assert opt.search_data["continuous"].nunique() > 1
    assert opt.search_data["categorical"].nunique() > 1


# =============================================================================
# Type correctness tests
# =============================================================================


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_continuous_returns_floats(optimizer_class):
    """Test that continuous dimensions return float values."""
    search_space = {
        "x": (-5.0, 5.0),
        "y": (0.0, 10.0),
    }

    def objective(params):
        return -(params["x"] ** 2 + params["y"] ** 2)

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=20, verbosity=[])

    assert isinstance(opt.best_para["x"], float)
    assert isinstance(opt.best_para["y"], float)
    assert -5.0 <= opt.best_para["x"] <= 5.0
    assert 0.0 <= opt.best_para["y"] <= 10.0


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_categorical_returns_original_values(optimizer_class):
    """Test that categorical dimensions return original category values."""
    search_space = {
        "optimizer": ["adam", "sgd", "rmsprop"],
        "use_bias": [True, False],
    }

    def objective(params):
        score = 0
        if params["optimizer"] == "adam":
            score += 0.5
        if params["use_bias"]:
            score += 0.1
        return score

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=20, verbosity=[])

    assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]
    assert opt.best_para["use_bias"] in [True, False]


# =============================================================================
# Edge cases
# =============================================================================


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_single_categorical_dimension(optimizer_class):
    """Test optimizer with only a single categorical dimension."""
    search_space = {"choice": ["a", "b", "c", "d", "e"]}

    def objective(params):
        return {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.2, "e": 0.4}[params["choice"]]

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=20, verbosity=[])

    assert opt.best_para["choice"] in ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_single_continuous_dimension(optimizer_class):
    """Test optimizer with only a single continuous dimension."""
    search_space = {"x": (0.0, 10.0)}

    def objective(params):
        return -((params["x"] - 5) ** 2)

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=20, verbosity=[])

    assert 0.0 <= opt.best_para["x"] <= 10.0


def test_categorical_with_none_values():
    """Test categorical dimension with None as a valid value."""
    search_space = {"option": [None, "enabled", "disabled"]}

    def objective(params):
        if params["option"] is None:
            return 0.5
        return 0.3

    opt = RandomSearchOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=20, verbosity=[])

    assert opt.best_para["option"] in [None, "enabled", "disabled"]


def test_categorical_with_numeric_values():
    """Test categorical dimension with numeric values (treated as categories)."""
    search_space = {"batch_size": [16, 32, 64, 128]}

    def objective(params):
        # Prefer 64
        return -abs(params["batch_size"] - 64)

    opt = HillClimbingOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=20, verbosity=[])

    assert opt.best_para["batch_size"] in [16, 32, 64, 128]


# =============================================================================
# Constraint handling
# =============================================================================


def test_constraint_on_continuous_dimension():
    """Test constraints work on continuous dimensions."""
    search_space = {"x": (-10.0, 10.0), "y": (-10.0, 10.0)}

    def constraint(params):
        return params["x"] > 0

    def objective(params):
        return -(params["x"] ** 2 + params["y"] ** 2)

    opt = RandomSearchOptimizer(search_space, constraints=[constraint], random_state=42)
    opt.search(objective, n_iter=50, verbosity=[])

    assert opt.best_para["x"] > 0


def test_constraint_on_categorical_dimension():
    """Test constraints work on categorical dimensions."""
    search_space = {
        "optimizer": ["adam", "sgd", "rmsprop", "adagrad"],
        "lr": (0.001, 0.1),
    }

    def constraint(params):
        return params["optimizer"] in ["adam", "sgd"]

    def objective(params):
        return -params["lr"]

    opt = RandomSearchOptimizer(search_space, constraints=[constraint], random_state=42)
    opt.search(objective, n_iter=50, verbosity=[])

    assert opt.best_para["optimizer"] in ["adam", "sgd"]


def test_constraint_on_mixed_dimensions():
    """Test constraints work across mixed dimension types."""
    search_space = {
        "x": np.arange(-5, 5),
        "y": (-5.0, 5.0),
        "algo": ["adam", "sgd"],
    }

    def constraint(params):
        return params["x"] + params["y"] > 0

    def objective(params):
        return -(params["x"] ** 2 + params["y"] ** 2)

    opt = HillClimbingOptimizer(search_space, constraints=[constraint], random_state=42)
    opt.search(objective, n_iter=50, verbosity=[])

    assert opt.best_para["x"] + opt.best_para["y"] > 0


# =============================================================================
# Reproducibility
# =============================================================================


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_reproducibility_with_random_state(optimizer_class):
    """Test that same random_state produces same results."""
    search_space = {
        "x": (0.0, 10.0),
        "category": ["a", "b", "c"],
    }

    def objective(params):
        score = -((params["x"] - 5) ** 2)
        if params["category"] == "b":
            score += 1
        return score

    kwargs = get_optimizer_kwargs(optimizer_class)

    opt1 = optimizer_class(search_space, random_state=42, **kwargs)
    opt1.search(objective, n_iter=20, verbosity=[])

    opt2 = optimizer_class(search_space, random_state=42, **kwargs)
    opt2.search(objective, n_iter=20, verbosity=[])

    assert opt1.best_score == opt2.best_score
    assert opt1.best_para == opt2.best_para

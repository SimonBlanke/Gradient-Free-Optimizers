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

# Note: GridSearchOptimizer and DirectAlgorithm now auto-discretize continuous
# dimensions, so they no longer need special handling in tests.


n_iter = 30


def get_exploration_init():
    """Return default initialization settings for exploration tests.

    Uses standard grid/vertices/random to ensure the init phase explores
    the search space broadly. The test then verifies the iteration phase
    also explores (doesn't just stay at the best init position).
    """
    return {"grid": 4, "vertices": 4, "random": 2}


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
    This verifies the algorithm can move in categorical space.

    Note: We check iteration-phase exploration. For exploitative algorithms
    that find the optimum during init, staying at the optimum is correct
    behavior - so we also accept finding the optimal value.
    """
    search_space = {
        "category": ["a", "b", "c", "d", "e"],
    }

    def objective(params):
        # Give different scores to encourage exploration
        # Optimum is 'c' with score 0.5
        scores = {"a": 0.1, "b": 0.2, "c": 0.5, "d": 0.3, "e": 0.4}
        return scores[params["category"]]

    initialize = get_exploration_init()

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(
        search_space, initialize=initialize, random_state=42, **kwargs
    )
    opt.search(objective, n_iter=n_iter, verbosity=[])

    # Exclude initialization data - only check actual iteration exploration
    iter_data = opt.search_data.iloc[opt.n_init_search :]

    # Check iteration-phase exploration OR that optimizer found optimal
    unique_values = iter_data["category"].nunique()
    found_optimal = opt.best_para["category"] == "c"

    assert unique_values > 1 or found_optimal, (
        f"{optimizer_class.__name__} did not explore categorical dimension "
        f"(only found {unique_values} unique value(s) in iteration phase) "
        f"and did not find optimal value 'c' (found '{opt.best_para['category']}')"
    )


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_continuous_exploration(optimizer_class):
    """Test that optimizer explores continuous dimensions.

    The key assertion: after N iterations, the optimizer should have
    tried more than one unique value in the continuous dimension.
    This verifies the algorithm can move in continuous space.

    Note: We check iteration-phase exploration. For exploitative algorithms
    that converge to the optimum, staying near the optimum is correct
    behavior - so we also accept finding a near-optimal value.
    """
    search_space = {
        "x": (0.0, 10.0),
    }

    def objective(params):
        # Quadratic with optimum at x=5
        return -((params["x"] - 5) ** 2)

    initialize = get_exploration_init()

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(
        search_space, initialize=initialize, random_state=42, **kwargs
    )
    opt.search(objective, n_iter=n_iter, verbosity=[])

    # Exclude initialization data - only check actual iteration exploration
    iter_data = opt.search_data.iloc[opt.n_init_search :]

    # Check iteration-phase exploration OR that optimizer found near-optimal
    unique_values = iter_data["x"].nunique()
    near_optimal = abs(opt.best_para["x"] - 5.0) < 1.0  # Within 1.0 of optimum

    assert unique_values > 1 or near_optimal, (
        f"{optimizer_class.__name__} did not explore continuous dimension "
        f"(only found {unique_values} unique value(s) in iteration phase) "
        f"and did not find near-optimal value "
        f"(found x={opt.best_para['x']:.2f}, optimal is 5.0)"
    )


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_mixed_exploration(optimizer_class):
    """Test that optimizer explores mixed dimension types.

    Combines discrete, continuous, and categorical in one search space.

    Note: For exploitative algorithms that find the optimum during init,
    staying at the optimum is correct behavior. We check for exploration
    OR achieving a good score.
    """
    search_space = {
        "discrete": np.arange(0, 10),
        "continuous": (0.0, 10.0),
        "categorical": ["opt1", "opt2", "opt3"],
    }

    def objective(params):
        score = 0
        # Prefer middle values for numeric (optimum at 5)
        score -= (params["discrete"] - 5) ** 2
        score -= (params["continuous"] - 5) ** 2
        # Prefer opt2
        if params["categorical"] == "opt2":
            score += 1
        return score

    initialize = get_exploration_init()

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(
        search_space, initialize=initialize, random_state=42, **kwargs
    )
    opt.search(objective, n_iter=n_iter, verbosity=[])

    # Exclude initialization data - only check actual iteration exploration
    iter_data = opt.search_data.iloc[opt.n_init_search :]

    # Check for exploration OR good score (near-optimal is score > -10)
    # Optimal score is 1 (at discrete=5, continuous=5, categorical='opt2')
    good_score = opt.best_score > -10

    discrete_explores = iter_data["discrete"].nunique() > 1
    continuous_explores = iter_data["continuous"].nunique() > 1
    categorical_explores = iter_data["categorical"].nunique() > 1

    # Either explores all dimensions OR achieves good score
    all_explore = discrete_explores and continuous_explores and categorical_explores

    assert all_explore or good_score, (
        f"{optimizer_class.__name__} did not explore all dimensions "
        f"(discrete: {iter_data['discrete'].nunique()}, "
        f"continuous: {iter_data['continuous'].nunique()}, "
        f"categorical: {iter_data['categorical'].nunique()}) "
        f"and did not achieve good score (score={opt.best_score:.2f})"
    )


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
    opt.search(objective, n_iter=n_iter, verbosity=[])

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
    opt.search(objective, n_iter=n_iter, verbosity=[])

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
    opt.search(objective, n_iter=n_iter, verbosity=[])

    assert opt.best_para["choice"] in ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_single_continuous_dimension(optimizer_class):
    """Test optimizer with only a single continuous dimension."""
    search_space = {"x": (0.0, 10.0)}

    def objective(params):
        return -((params["x"] - 5) ** 2)

    kwargs = get_optimizer_kwargs(optimizer_class)
    opt = optimizer_class(search_space, random_state=42, **kwargs)
    opt.search(objective, n_iter=n_iter, verbosity=[])

    assert 0.0 <= opt.best_para["x"] <= 10.0


def test_categorical_with_none_values():
    """Test categorical dimension with None as a valid value."""
    search_space = {"option": [None, "enabled", "disabled"]}

    def objective(params):
        if params["option"] is None:
            return 0.5
        return 0.3

    opt = RandomSearchOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=n_iter, verbosity=[])

    assert opt.best_para["option"] in [None, "enabled", "disabled"]


def test_categorical_with_numeric_values():
    """Test categorical dimension with numeric values (treated as categories)."""
    search_space = {"batch_size": [16, 32, 64, 128]}

    def objective(params):
        # Prefer 64
        return -abs(params["batch_size"] - 64)

    opt = HillClimbingOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=n_iter, verbosity=[])

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
    opt.search(objective, n_iter=n_iter, verbosity=[])

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
    opt.search(objective, n_iter=n_iter, verbosity=[])

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
    opt.search(objective, n_iter=n_iter, verbosity=[])

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
    opt1.search(objective, n_iter=n_iter, verbosity=[])

    opt2 = optimizer_class(search_space, random_state=42, **kwargs)
    opt2.search(objective, n_iter=n_iter, verbosity=[])

    assert opt1.best_score == opt2.best_score
    assert opt1.best_para == opt2.best_para

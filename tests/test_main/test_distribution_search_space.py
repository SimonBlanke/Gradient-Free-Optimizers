"""Tests for SciPy distribution-backed search space dimensions."""

import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.scipy

stats = pytest.importorskip("scipy.stats")

from gradient_free_optimizers import (
    BayesianOptimizer,
    CMAESOptimizer,
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
from gradient_free_optimizers._dimension_types import (
    DEFAULT_DISTRIBUTION_QUANTILES,
    DimensionType,
    classify_search_space_value,
    distribution_quantile_bounds,
)
from gradient_free_optimizers.optimizers.core_optimizer import Converter

DISTRIBUTION_OPTIMIZERS = [
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
    CMAESOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
]


POPULATION_OPTIMIZERS = {
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer,
    CMAESOptimizer,
}


def _optimizer_kwargs(optimizer_class):
    if optimizer_class in POPULATION_OPTIMIZERS:
        return {"population": 5}
    return {}


SCIPY_DISTRIBUTIONS = [
    pytest.param(stats.uniform(loc=0, scale=10), (0.0, 1.0), id="uniform"),
    pytest.param(stats.beta(2, 5), (0.0, 1.0), id="beta"),
    pytest.param(stats.loguniform(1e-5, 1e-1), (0.0, 1.0), id="loguniform"),
    pytest.param(stats.truncnorm(-2, 2, loc=0, scale=1), (0.0, 1.0), id="truncnorm"),
    pytest.param(stats.norm(loc=0, scale=1), DEFAULT_DISTRIBUTION_QUANTILES, id="norm"),
    pytest.param(stats.t(df=5), DEFAULT_DISTRIBUTION_QUANTILES, id="t"),
    pytest.param(
        stats.laplace(loc=0, scale=1), DEFAULT_DISTRIBUTION_QUANTILES, id="laplace"
    ),
    pytest.param(stats.cauchy(), DEFAULT_DISTRIBUTION_QUANTILES, id="cauchy"),
    pytest.param(stats.expon(scale=1), DEFAULT_DISTRIBUTION_QUANTILES, id="expon"),
    pytest.param(stats.lognorm(s=1), DEFAULT_DISTRIBUTION_QUANTILES, id="lognorm"),
    pytest.param(stats.chi2(df=3), DEFAULT_DISTRIBUTION_QUANTILES, id="chi2"),
    pytest.param(stats.halfnorm(), DEFAULT_DISTRIBUTION_QUANTILES, id="halfnorm"),
]


@pytest.mark.parametrize(("dist", "expected_bounds"), SCIPY_DISTRIBUTIONS)
def test_distribution_detection_and_bounds(dist, expected_bounds):
    assert classify_search_space_value(dist) == DimensionType.DISTRIBUTION
    assert distribution_quantile_bounds(dist) == expected_bounds


@pytest.mark.parametrize(("dist", "expected_bounds"), SCIPY_DISTRIBUTIONS)
def test_distribution_converter_roundtrip_parametrized(dist, expected_bounds):
    conv = Converter({"x": dist})

    median = float(dist.ppf(0.5))
    pos = conv.value2position([median])
    assert abs(float(pos[0]) - 0.5) < 1e-6

    val = conv.position2value([0.5])
    assert abs(val[0] - median) < 1e-6


@pytest.mark.parametrize(("dist", "expected_bounds"), SCIPY_DISTRIBUTIONS)
def test_distribution_optimization_across_distributions(dist, expected_bounds):
    search_space = {"x": dist}
    median = float(dist.ppf(0.5))

    def objective(params):
        return -abs(params["x"] - median)

    opt = HillClimbingOptimizer(search_space, initialize={"random": 3}, random_state=42)
    opt.search(objective, n_iter=15, verbosity=False)

    q_low, q_high = expected_bounds
    low_value = float(dist.ppf(q_low))
    high_value = float(dist.ppf(q_high))

    assert math.isfinite(opt.best_para["x"])
    assert low_value <= opt.best_para["x"] <= high_value


def test_distribution_converter_roundtrip_in_quantile_space():
    conv = Converter({"x": stats.norm(loc=0, scale=1)})

    assert conv.dim_types == [DimensionType.DISTRIBUTION]
    assert abs(conv.position2value([0.5])[0]) < 1e-12
    assert abs(float(conv.value2position([0.0])[0]) - 0.5) < 1e-12

    # Infinite-support distributions are clipped to effective quantile bounds.
    lower_value = conv.position2value([0.0])[0]
    assert math.isfinite(lower_value)
    assert lower_value == stats.norm().ppf(DEFAULT_DISTRIBUTION_QUANTILES[0])


def test_distribution_nan_position_uses_mid_quantile():
    conv = Converter({"x": stats.norm(loc=0, scale=1)})

    value = conv.position2value([float("nan")])

    assert math.isfinite(value[0])
    assert abs(value[0]) < 1e-12


def test_distribution_values2positions_preserves_float_positions():
    conv = Converter({"x": stats.norm(loc=0, scale=1)})

    positions = conv.values2positions([[0.0], [stats.norm().ppf(0.75)]])

    assert abs(float(positions[0][0]) - 0.5) < 1e-12
    assert abs(float(positions[1][0]) - 0.75) < 1e-12


def test_distribution_memory_roundtrip_uses_quantile_keys():
    conv = Converter({"x": stats.norm(loc=0, scale=1)})
    search_data = pd.DataFrame({"x": [0.0], "score": [1.0]})

    memory = conv.dataframe2memory_dict(search_data)
    key = next(iter(memory.keys()))
    restored = conv.memory_dict2dataframe(memory)

    assert abs(key[0] - 0.5) < 1e-12
    assert abs(restored["x"].iloc[0]) < 1e-12
    assert restored["score"].iloc[0] == 1.0


def test_distribution_memory_warm_start_drops_clipped_tail_values(caplog):
    dist = stats.norm(loc=0, scale=1)
    conv = Converter({"x": dist})
    upper_edge = float(dist.ppf(DEFAULT_DISTRIBUTION_QUANTILES[1]))
    search_data = pd.DataFrame({"x": [0.0, upper_edge, 4.0], "score": [0.0, 1.0, 4.0]})

    with caplog.at_level("WARNING"):
        memory = conv.dataframe2memory_dict(search_data)

    restored = conv.memory_dict2dataframe(memory)

    assert len(restored) == 2
    assert set(restored["score"]) == {0.0, 1.0}
    assert "Dropped 1 memory warm-start rows" in caplog.text


def test_distribution_memory_warm_start_drops_values_outside_finite_support(caplog):
    conv = Converter({"x": stats.uniform(loc=0, scale=10)})
    search_data = pd.DataFrame(
        {"x": [-1.0, 0.0, 10.0, 11.0], "score": [-1.0, 0.0, 10.0, 11.0]}
    )

    with caplog.at_level("WARNING"):
        memory = conv.dataframe2memory_dict(search_data)

    restored = conv.memory_dict2dataframe(memory)

    assert len(restored) == 2
    assert set(restored["score"]) == {0.0, 10.0}
    assert all(0.0 <= value <= 10.0 for value in restored["x"])
    assert "Dropped 2 memory warm-start rows" in caplog.text


def test_random_search_samples_distribution_values():
    search_space = {"lr": stats.loguniform(1e-5, 1e-1)}

    def objective(params):
        return -abs(math.log10(params["lr"]) + 3)

    opt = RandomSearchOptimizer(search_space, initialize={"random": 3}, random_state=1)
    opt.search(objective, n_iter=10, verbosity=False)

    values = list(opt.search_data["lr"])

    assert opt._distribution_mask.sum() == 1
    assert opt._continuous_mask.sum() == 1
    assert all(1e-5 <= value <= 1e-1 for value in values)


@pytest.mark.parametrize("optimizer_class", DISTRIBUTION_OPTIMIZERS)
def test_distribution_dimension_runs_for_all_optimizers(optimizer_class):
    search_space = {"x": stats.norm(loc=0, scale=1)}

    def objective(params):
        return -abs(params["x"])

    opt = optimizer_class(
        search_space,
        initialize={"random": 3},
        random_state=2,
        **_optimizer_kwargs(optimizer_class),
    )
    opt.search(objective, n_iter=8, verbosity=False)

    lower = stats.norm().ppf(DEFAULT_DISTRIBUTION_QUANTILES[0])
    upper = stats.norm().ppf(DEFAULT_DISTRIBUTION_QUANTILES[1])

    assert math.isfinite(opt.best_para["x"])
    assert lower <= opt.best_para["x"] <= upper


def test_mixed_search_space_all_four_dimension_types():
    search_space = {
        "lr": stats.loguniform(1e-5, 1e-1),
        "epochs": np.arange(10, 110, 10),
        "optimizer": ["adam", "sgd", "rmsprop"],
        "dropout": (0.0, 0.5),
    }

    def objective(params):
        score = -abs(math.log10(params["lr"]) + 3)
        score -= abs(params["epochs"] - 50) / 100
        score -= 0.1 if params["optimizer"] != "adam" else 0
        score -= abs(params["dropout"] - 0.2)
        return score

    opt = RandomSearchOptimizer(search_space, initialize={"random": 4}, random_state=0)
    opt.search(objective, n_iter=20, verbosity=False)

    assert 1e-5 <= opt.best_para["lr"] <= 1e-1
    assert opt.best_para["epochs"] in range(10, 110, 10)
    assert opt.best_para["optimizer"] in ["adam", "sgd", "rmsprop"]
    assert 0.0 <= opt.best_para["dropout"] <= 0.5
    assert len(opt.search_data) == 20


def test_distribution_with_constraints():
    search_space = {"x": stats.norm(loc=0, scale=1)}

    def objective(params):
        return -abs(params["x"] - 0.5)

    opt = HillClimbingOptimizer(
        search_space,
        constraints=[lambda p: p["x"] > 0],
        initialize={"random": 3},
        random_state=42,
    )
    opt.search(objective, n_iter=15, verbosity=False)

    assert all(row["x"] > 0 for _, row in opt.search_data.iterrows())


def test_distribution_grid_initialization():
    search_space = {"x": stats.uniform(loc=0, scale=10)}

    def objective(params):
        return -abs(params["x"] - 5)

    opt = HillClimbingOptimizer(
        search_space,
        initialize={"grid": 5},
        random_state=1,
    )
    opt.search(objective, n_iter=10, verbosity=False)

    assert len(opt.search_data) == 10
    assert all(0 <= row["x"] <= 10 for _, row in opt.search_data.iterrows())


def test_distribution_vertex_initialization():
    search_space = {"x": stats.uniform(loc=0, scale=10)}

    def objective(params):
        return -abs(params["x"] - 5)

    opt = HillClimbingOptimizer(
        search_space,
        initialize={"vertices": 2},
        random_state=1,
    )
    opt.search(objective, n_iter=10, verbosity=False)

    assert len(opt.search_data) == 10
    first_values = list(opt.search_data["x"].iloc[:2])
    assert any(v <= 0.01 for v in first_values) or any(v >= 9.99 for v in first_values)


def test_distribution_ask_tell_interface():
    from gradient_free_optimizers.ask_tell import HillClimbingOptimizer as AskTellHC

    search_space = {"x": stats.norm(loc=0, scale=1)}

    def objective(params):
        return -abs(params["x"])

    initial_evals = [
        ({"x": 0.0}, objective({"x": 0.0})),
        ({"x": 0.5}, objective({"x": 0.5})),
    ]
    opt = AskTellHC(search_space, initial_evaluations=initial_evals, random_state=0)

    for _ in range(10):
        params_list = opt.ask(n=1)
        assert len(params_list) == 1
        params = params_list[0]
        assert "x" in params
        assert isinstance(params["x"], float)
        assert math.isfinite(params["x"])
        scores = [objective(params)]
        opt.tell(scores)

    assert math.isfinite(opt.best_score)
    assert math.isfinite(opt.best_para["x"])


def test_smbo_warm_start_with_distribution():
    search_space = {"x": stats.norm(loc=0, scale=1)}

    previous_data = pd.DataFrame(
        {"x": [0.0, 0.5, -0.5, 1.0, -1.0], "score": [0.0, -0.5, -0.5, -1.0, -1.0]}
    )

    def objective(params):
        return -abs(params["x"])

    opt = BayesianOptimizer(
        search_space,
        initialize={"random": 2},
        random_state=1,
        warm_start_smbo=previous_data,
    )
    opt.search(
        objective,
        n_iter=10,
        verbosity=False,
    )

    assert len(opt.search_data) == 10
    assert math.isfinite(opt.best_score)


def test_smbo_warm_start_drops_invalid_distribution_rows(caplog):
    dist = stats.norm(loc=0, scale=1)
    previous_data = pd.DataFrame(
        {
            "x": [0.0, float(dist.ppf(DEFAULT_DISTRIBUTION_QUANTILES[1])), 4.0],
            "score": [0.0, -1.0, -4.0],
        }
    )

    with caplog.at_level("WARNING"):
        opt = BayesianOptimizer(
            {"x": dist},
            initialize={"random": 2},
            random_state=1,
            warm_start_smbo=previous_data,
        )

    assert len(opt.X_sample) == 2
    assert opt.Y_sample == [0.0, -1.0]
    assert "Dropped 1 SMBO warm-start rows" in caplog.text

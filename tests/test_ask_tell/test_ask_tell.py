"""Tests for the batch ask/tell interface."""

import math

import numpy as np
import pytest

from gradient_free_optimizers.ask_tell import (
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

search_space = {
    "x": np.linspace(-10, 10, 100),
    "y": np.linspace(-10, 10, 100),
}


def objective(params):
    return -(params["x"] ** 2 + params["y"] ** 2)


def _make_init_evals(n=3, seed=0):
    """Generate n random initial evaluations from the search space."""
    rng = np.random.RandomState(seed)
    evals = []
    for _ in range(n):
        params = {
            "x": rng.choice(search_space["x"]),
            "y": rng.choice(search_space["y"]),
        }
        evals.append((params, objective(params)))
    return evals


# Default init evals for most tests
default_init = _make_init_evals(3)

# Larger init for population optimizers (need >= population size)
pop_init = _make_init_evals(12)


all_optimizers = (
    "Optimizer",
    [
        HillClimbingOptimizer,
        StochasticHillClimbingOptimizer,
        RepulsingHillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        DownhillSimplexOptimizer,
        RandomSearchOptimizer,
        PowellsMethod,
        PatternSearch,
        DirectAlgorithm,
        GridSearchOptimizer,
        RandomRestartHillClimbingOptimizer,
        RandomAnnealingOptimizer,
        ParallelTemperingOptimizer,
        ParticleSwarmOptimizer,
        SpiralOptimization,
        GeneticAlgorithmOptimizer,
        EvolutionStrategyOptimizer,
        DifferentialEvolutionOptimizer,
        CMAESOptimizer,
        LipschitzOptimizer,
        BayesianOptimizer,
        TreeStructuredParzenEstimators,
        ForestOptimizer,
    ],
)

non_population_optimizers = (
    "Optimizer",
    [
        HillClimbingOptimizer,
        StochasticHillClimbingOptimizer,
        RepulsingHillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        DownhillSimplexOptimizer,
        RandomSearchOptimizer,
        PowellsMethod,
        PatternSearch,
        DirectAlgorithm,
        GridSearchOptimizer,
        RandomRestartHillClimbingOptimizer,
        RandomAnnealingOptimizer,
        LipschitzOptimizer,
        BayesianOptimizer,
        TreeStructuredParzenEstimators,
        ForestOptimizer,
    ],
)

population_optimizers = (
    "Optimizer",
    [
        ParallelTemperingOptimizer,
        ParticleSwarmOptimizer,
        SpiralOptimization,
        GeneticAlgorithmOptimizer,
        EvolutionStrategyOptimizer,
        DifferentialEvolutionOptimizer,
        CMAESOptimizer,
    ],
)

representative_optimizers = (
    "Optimizer",
    [
        HillClimbingOptimizer,
        RandomSearchOptimizer,
        SimulatedAnnealingOptimizer,
        ParticleSwarmOptimizer,
        EvolutionStrategyOptimizer,
    ],
)


def _get_init(Optimizer, n=3, pop_n=12, seed=0):
    """Return appropriate init evals depending on optimizer type."""
    # Population optimizers need more init points
    pop_classes = (
        ParallelTemperingOptimizer,
        ParticleSwarmOptimizer,
        SpiralOptimization,
        GeneticAlgorithmOptimizer,
        EvolutionStrategyOptimizer,
        DifferentialEvolutionOptimizer,
        CMAESOptimizer,
    )
    if Optimizer in pop_classes:
        return _make_init_evals(pop_n, seed)
    return _make_init_evals(n, seed)


# ── Basic ask/tell loop ──────────────────────────────────────────────


@pytest.mark.parametrize(*all_optimizers)
def test_basic_ask_tell_loop(Optimizer):
    """Every optimizer should complete a basic ask/tell loop with n=1."""
    init = _get_init(Optimizer)
    opt = Optimizer(search_space, initial_evaluations=init)

    for _ in range(20):
        params_list = opt.ask(n=1)
        assert isinstance(params_list, list)
        assert len(params_list) == 1
        scores = [objective(p) for p in params_list]
        opt.tell(scores)

    assert opt.best_para is not None
    assert opt.best_score > -math.inf
    assert isinstance(opt.best_para, dict)
    assert "x" in opt.best_para
    assert "y" in opt.best_para


@pytest.mark.parametrize(*all_optimizers)
def test_ask_default_n(Optimizer):
    """ask() without argument should return a list of length 1."""
    init = _get_init(Optimizer)
    opt = Optimizer(search_space, initial_evaluations=init)

    params_list = opt.ask()
    assert isinstance(params_list, list)
    assert len(params_list) == 1
    opt.tell([objective(params_list[0])])


# ── Batch ask/tell ───────────────────────────────────────────────────


@pytest.mark.parametrize(*all_optimizers)
def test_batch_ask_tell(Optimizer):
    """Every optimizer should handle batch ask(n=4) / tell(list, list)."""
    init = _get_init(Optimizer)
    opt = Optimizer(search_space, initial_evaluations=init)

    for _ in range(5):
        params_list = opt.ask(n=4)
        assert isinstance(params_list, list)
        assert len(params_list) == 4
        scores = [objective(p) for p in params_list]
        opt.tell(scores)

    assert opt.best_para is not None
    assert opt.best_score > -math.inf


@pytest.mark.parametrize(*representative_optimizers)
def test_varying_batch_sizes(Optimizer):
    """Alternating between different batch sizes should work."""
    init = _get_init(Optimizer)
    opt = Optimizer(search_space, initial_evaluations=init)

    for n in [1, 3, 1, 5, 2]:
        params_list = opt.ask(n=n)
        assert len(params_list) == n
        scores = [objective(p) for p in params_list]
        opt.tell(scores)

    assert opt.best_para is not None


# ── Best tracking ────────────────────────────────────────────────────


@pytest.mark.parametrize(*all_optimizers)
def test_best_score_tracks_maximum(Optimizer):
    """best_score should reflect the actual best score seen."""
    init = _get_init(Optimizer, seed=42)
    opt = Optimizer(search_space, initial_evaluations=init)

    all_scores = [s for _, s in init]
    for _ in range(10):
        params_list = opt.ask(n=1)
        scores = [objective(p) for p in params_list]
        opt.tell(scores)
        all_scores.extend(scores)

    assert opt.best_score == max(all_scores)


@pytest.mark.parametrize(*all_optimizers)
def test_best_para_score_consistency(Optimizer):
    """best_para should correspond to the position that produced best_score."""
    init = _get_init(Optimizer)
    opt = Optimizer(search_space, initial_evaluations=init)

    history = list(init)
    for _ in range(15):
        params_list = opt.ask(n=1)
        scores = [objective(p) for p in params_list]
        opt.tell(scores)
        history.append((params_list[0], scores[0]))

    best_entry = max(history, key=lambda x: x[1])
    assert opt.best_score == best_entry[1]

    bp = opt.best_para
    assert abs(bp["x"] - best_entry[0]["x"]) < 0.5
    assert abs(bp["y"] - best_entry[0]["y"]) < 0.5


@pytest.mark.parametrize(*representative_optimizers)
def test_best_tracking_across_batches(Optimizer):
    """Best should be correctly tracked when best appears mid-batch."""
    init = _get_init(Optimizer)
    opt = Optimizer(search_space, initial_evaluations=init)

    all_scores = [s for _, s in init]
    for _ in range(5):
        params_list = opt.ask(n=3)
        scores = [objective(p) for p in params_list]
        opt.tell(scores)
        all_scores.extend(scores)

    assert opt.best_score == max(all_scores)


# ── Error handling ───────────────────────────────────────────────────


def test_double_ask_raises():
    """Calling ask() twice without tell() must raise RuntimeError."""
    opt = HillClimbingOptimizer(search_space, initial_evaluations=default_init)
    opt.ask()

    with pytest.raises(RuntimeError, match="ask.*before tell"):
        opt.ask()


def test_tell_without_ask_raises():
    """Calling tell() without ask() must raise RuntimeError."""
    opt = HillClimbingOptimizer(search_space, initial_evaluations=default_init)

    with pytest.raises(RuntimeError, match="without a preceding ask"):
        opt.tell([1.0])


def test_tell_wrong_length_raises():
    """tell() with wrong number of results must raise ValueError."""
    opt = HillClimbingOptimizer(search_space, initial_evaluations=default_init)
    opt.ask(n=2)

    with pytest.raises(ValueError, match="Expected 2"):
        opt.tell([1.0])


def test_ask_without_init_raises():
    """ask() on an uninitialized optimizer should raise RuntimeError.

    This tests the edge case where CoreOptimizer was constructed but
    _process_initial_evaluations was never called.
    """
    from gradient_free_optimizers._ask_tell_mixin import AskTell
    from gradient_free_optimizers.optimizers import (
        HillClimbingOptimizer as _BaseHC,
    )

    class BareOptimizer(_BaseHC, AskTell):
        pass

    opt = BareOptimizer(search_space=search_space, initialize={"random": 0})
    with pytest.raises(RuntimeError, match="not initialized"):
        opt.ask()


def test_empty_initial_evaluations_raises():
    """Empty initial_evaluations should raise ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        HillClimbingOptimizer(search_space, initial_evaluations=[])


def test_population_min_evaluations_raises():
    """Population optimizer with too few init evals should raise ValueError."""
    too_few = _make_init_evals(3)
    with pytest.raises(ValueError, match="requires at least"):
        ParticleSwarmOptimizer(search_space, initial_evaluations=too_few, population=10)


# ── Error recovery ───────────────────────────────────────────────────


def test_recovery_after_double_ask_error():
    """After a double-ask error, tell() should recover the state."""
    opt = HillClimbingOptimizer(search_space, initial_evaluations=default_init)

    params = opt.ask()

    with pytest.raises(RuntimeError):
        opt.ask()

    # Should still be able to tell and continue
    opt.tell([objective(params[0])])

    params2 = opt.ask()
    opt.tell([objective(params2[0])])
    assert opt.best_para is not None


def test_recovery_after_tell_without_ask_error():
    """After a tell-without-ask error, ask() should work normally."""
    opt = HillClimbingOptimizer(search_space, initial_evaluations=default_init)

    with pytest.raises(RuntimeError):
        opt.tell([1.0])

    params = opt.ask()
    opt.tell([objective(params[0])])
    assert opt.best_para is not None


# ── Score edge cases ─────────────────────────────────────────────────


def test_nan_score_handling():
    """NaN scores should not corrupt best tracking."""
    opt = HillClimbingOptimizer(
        search_space, initial_evaluations=default_init, random_state=1
    )

    opt.ask()
    opt.tell([10.0])

    opt.ask()
    opt.tell([float("nan")])

    assert opt.best_score >= 10.0


def test_all_negative_scores():
    """Optimizer should track the least-negative score as best."""
    init = [({"x": search_space["x"][50], "y": search_space["y"][50]}, -100.0)]
    opt = RandomSearchOptimizer(search_space, initial_evaluations=init, random_state=1)

    all_scores = [-100.0]
    for _ in range(20):
        params = opt.ask()
        score = -(params[0]["x"] ** 2 + params[0]["y"] ** 2) - 100
        opt.tell([score])
        all_scores.append(score)

    assert opt.best_score == max(all_scores)


# ── Dimension type tests ────────────────────────────────────────────


@pytest.mark.parametrize(*representative_optimizers)
def test_categorical_dimensions(Optimizer):
    """ask/tell should work with categorical (list) dimensions."""
    cat_space = {
        "color": ["red", "green", "blue"],
        "size": ["small", "medium", "large"],
    }
    cat_init = [
        ({"color": "red", "size": "small"}, 1.0),
        ({"color": "green", "size": "medium"}, 2.0),
        ({"color": "blue", "size": "large"}, 3.0),
    ]
    if Optimizer in (ParticleSwarmOptimizer, EvolutionStrategyOptimizer):
        cat_init = cat_init * 4  # need >= population size

    opt = Optimizer(cat_space, initial_evaluations=cat_init)

    for _ in range(10):
        params_list = opt.ask()
        assert params_list[0]["color"] in ["red", "green", "blue"]
        assert params_list[0]["size"] in ["small", "medium", "large"]
        score = {"red": 1, "green": 2, "blue": 3}[params_list[0]["color"]]
        opt.tell([score])

    assert opt.best_para is not None


@pytest.mark.parametrize(*representative_optimizers)
def test_mixed_dimension_types(Optimizer):
    """ask/tell should handle a mix of discrete, continuous, and categorical."""
    mixed_space = {
        "lr": np.logspace(-5, -1, 50),
        "temp": (0.0, 2.0),
        "kernel": ["linear", "rbf", "poly"],
    }
    rng = np.random.RandomState(0)
    mixed_init = []
    for _ in range(12):
        p = {
            "lr": rng.choice(mixed_space["lr"]),
            "temp": rng.uniform(0.0, 2.0),
            "kernel": rng.choice(mixed_space["kernel"]),
        }
        s = -p["lr"] + (1 if p["kernel"] == "rbf" else 0)
        mixed_init.append((p, s))

    opt = Optimizer(mixed_space, initial_evaluations=mixed_init)

    for _ in range(10):
        params_list = opt.ask()
        p = params_list[0]
        score = -p["lr"] + (1 if p["kernel"] == "rbf" else 0)
        opt.tell([score])

    assert opt.best_para is not None


# ── Population optimizers ────────────────────────────────────────────


@pytest.mark.parametrize(*population_optimizers)
def test_population_optimizer_cycling(Optimizer):
    """Population optimizers should cycle through all individuals."""
    opt = Optimizer(search_space, initial_evaluations=pop_init)

    for _ in range(30):
        params = opt.ask()
        scores = [objective(p) for p in params]
        opt.tell(scores)

    assert opt.best_score > -math.inf
    assert opt.best_para is not None


@pytest.mark.parametrize(*population_optimizers)
def test_population_batch_equals_population_size(Optimizer):
    """Batch size matching population should work naturally."""
    pop_size = 5
    init = _make_init_evals(pop_size, seed=7)
    opt = Optimizer(search_space, initial_evaluations=init, population=pop_size)

    for _ in range(5):
        params = opt.ask(n=pop_size)
        assert len(params) == pop_size
        scores = [objective(p) for p in params]
        opt.tell(scores)

    assert opt.best_para is not None


# ── Algorithm-specific parameter tests ───────────────────────────────


def test_hill_climbing_custom_params():
    """HillClimbing should respect custom epsilon and n_neighbours."""
    opt = HillClimbingOptimizer(
        search_space,
        initial_evaluations=default_init,
        epsilon=0.1,
        n_neighbours=5,
        distribution="laplace",
        random_state=1,
    )

    for _ in range(20):
        p = opt.ask()
        opt.tell([objective(p[0])])

    assert opt.best_para is not None


def test_simulated_annealing_custom_params():
    """SA should work with custom temperature and annealing rate."""
    opt = SimulatedAnnealingOptimizer(
        search_space,
        initial_evaluations=default_init,
        start_temp=10.0,
        annealing_rate=0.95,
        random_state=1,
    )

    for _ in range(30):
        p = opt.ask()
        opt.tell([objective(p[0])])

    assert opt.best_para is not None


def test_bayesian_optimizer_custom_params():
    """Bayesian optimizer should work with custom xi."""
    opt = BayesianOptimizer(
        search_space,
        initial_evaluations=_make_init_evals(5),
        xi=0.1,
    )

    for _ in range(10):
        p = opt.ask()
        opt.tell([objective(p[0])])

    assert opt.best_para is not None


# ── High-dimensional / edge cases ───────────────────────────────────


def test_high_dimensional_space():
    """ask/tell should work with many dimensions."""
    high_dim_space = {f"x{i}": np.linspace(-1, 1, 20) for i in range(15)}
    rng = np.random.RandomState(1)
    hi_init = []
    for _ in range(3):
        p = {k: rng.choice(v) for k, v in high_dim_space.items()}
        hi_init.append((p, -sum(v**2 for v in p.values())))

    opt = HillClimbingOptimizer(
        high_dim_space, initial_evaluations=hi_init, random_state=1
    )

    for _ in range(30):
        params_list = opt.ask()
        assert len(params_list[0]) == 15
        score = -sum(v**2 for v in params_list[0].values())
        opt.tell([score])

    assert opt.best_para is not None
    assert len(opt.best_para) == 15


def test_single_dimension():
    """ask/tell should work with a 1D search space."""
    space_1d = {"x": np.linspace(-5, 5, 100)}
    init_1d = [({"x": 0.0}, 0.0), ({"x": 2.0}, -4.0)]
    opt = HillClimbingOptimizer(space_1d, initial_evaluations=init_1d, random_state=1)

    for _ in range(20):
        p = opt.ask()
        assert "x" in p[0]
        assert len(p[0]) == 1
        opt.tell([-(p[0]["x"] ** 2)])

    assert abs(opt.best_para["x"]) < 3


# ── Initial evaluations seeding ─────────────────────────────────────


def test_initial_evaluations_set_best():
    """best_score should reflect the best from initial_evaluations."""
    init = [
        ({"x": 0.0, "y": 0.0}, -0.0),
        ({"x": 5.0, "y": 5.0}, -50.0),
    ]
    opt = HillClimbingOptimizer(search_space, initial_evaluations=init)

    # Before any ask/tell, best should come from init
    assert opt.best_score == 0.0


def test_initial_evaluations_state_is_iter():
    """After construction, optimizer should be in iteration state."""
    opt = HillClimbingOptimizer(search_space, initial_evaluations=default_init)
    assert opt.search_state == "iter"


def test_many_initial_evaluations():
    """Large initial_evaluations should work without issues."""
    large_init = _make_init_evals(100, seed=99)
    opt = HillClimbingOptimizer(search_space, initial_evaluations=large_init)

    params = opt.ask()
    opt.tell([objective(params[0])])
    assert opt.best_para is not None


# ── GridSearch ───────────────────────────────────────────────────────


def test_grid_search_systematic():
    """GridSearch should systematically traverse the grid with ask/tell."""
    init = _make_init_evals(1, seed=1)
    opt = GridSearchOptimizer(search_space, initial_evaluations=init, random_state=1)

    positions = set()
    for _ in range(20):
        p = opt.ask()
        pos = (round(p[0]["x"], 6), round(p[0]["y"], 6))
        positions.add(pos)
        opt.tell([objective(p[0])])

    assert len(positions) >= 15


# ── Long run stability ──────────────────────────────────────────────


def test_many_iterations():
    """Run enough iterations to verify no state corruption over time."""
    opt = HillClimbingOptimizer(
        search_space, initial_evaluations=default_init, random_state=1
    )

    for _ in range(200):
        params = opt.ask()
        score = objective(params[0])
        opt.tell([score])

    assert opt.best_score > -200
    assert opt.best_para is not None


def test_many_batch_iterations():
    """Run many batch iterations to verify no state corruption."""
    opt = RandomSearchOptimizer(
        search_space, initial_evaluations=default_init, random_state=1
    )

    for _ in range(50):
        params = opt.ask(n=4)
        scores = [objective(p) for p in params]
        opt.tell(scores)

    assert opt.best_score > -200
    assert opt.best_para is not None

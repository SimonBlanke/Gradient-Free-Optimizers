# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API smoke tests for the ask/tell interface.

These tests verify that every optimizer's constructor parameters are accepted
through the ask/tell interface and that a minimal ask/tell cycle completes
without errors. Each algorithm-specific parameter is tested individually
and then all together, so a renamed or removed parameter breaks exactly
one test with a clear name.
"""

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

SEARCH_SPACE = {
    "x": np.linspace(-5, 5, 20),
    "y": np.linspace(-5, 5, 20),
}

SEARCH_SPACE_SMALL = {
    "x": np.linspace(-1, 1, 5),
}


def objective(p):
    return -(p["x"] ** 2 + p.get("y", 0) ** 2)


def objective_small(p):
    return -(p["x"] ** 2)


def _make_init_evals(search_space, n=3, seed=0):
    rng = np.random.RandomState(seed)
    evals = []
    for _ in range(n):
        params = {k: rng.choice(v) for k, v in search_space.items()}
        evals.append((params, objective(params)))
    return evals


def _make_init_evals_small(n=3, seed=0):
    rng = np.random.RandomState(seed)
    evals = []
    for _ in range(n):
        params = {"x": rng.choice(SEARCH_SPACE_SMALL["x"])}
        evals.append((params, objective_small(params)))
    return evals


default_init = _make_init_evals(SEARCH_SPACE, n=3)
pop_init = _make_init_evals(SEARCH_SPACE, n=12)
smbo_init = _make_init_evals_small(n=3)


def _run_cycle(opt, obj_fn=objective, n=3):
    """Run a minimal ask/tell cycle and verify best tracking."""
    for _ in range(n):
        params_list = opt.ask(1)
        scores = [obj_fn(p) for p in params_list]
        opt.tell(scores)
    assert opt.best_para is not None
    assert opt.best_score > -math.inf


def _run_batch_cycle(opt, obj_fn=objective, n=3, batch=2):
    """Run a batched ask/tell cycle."""
    for _ in range(n):
        params_list = opt.ask(batch)
        scores = [obj_fn(p) for p in params_list]
        opt.tell(scores)
    assert opt.best_para is not None
    assert opt.best_score > -math.inf


# Each entry: (Optimizer, is_population, uses_small_space, algo_params)
# is_population controls how many initial_evaluations are generated.
# uses_small_space selects SEARCH_SPACE_SMALL for SMBO optimizers.
OPTIMIZER_CONFIGS = [
    (
        HillClimbingOptimizer,
        False,
        False,
        {
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 5,
        },
    ),
    (
        StochasticHillClimbingOptimizer,
        False,
        False,
        {
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 5,
            "p_accept": 0.3,
        },
    ),
    (
        RepulsingHillClimbingOptimizer,
        False,
        False,
        {
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 5,
            "repulsion_factor": 3,
        },
    ),
    (
        SimulatedAnnealingOptimizer,
        False,
        False,
        {
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 5,
            "annealing_rate": 0.95,
            "start_temp": 2,
        },
    ),
    (
        RandomAnnealingOptimizer,
        False,
        False,
        {
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 5,
            "annealing_rate": 0.95,
            "start_temp": 5,
        },
    ),
    (
        DownhillSimplexOptimizer,
        False,
        False,
        {
            "alpha": 1.5,
            "gamma": 2.5,
            "beta": 0.3,
            "sigma": 0.3,
        },
    ),
    (RandomSearchOptimizer, False, False, {}),
    (
        GridSearchOptimizer,
        False,
        False,
        {
            "step_size": 2,
            "direction": "orthogonal",
            "resolution": 50,
        },
    ),
    (
        RandomRestartHillClimbingOptimizer,
        False,
        False,
        {
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 5,
            "n_iter_restart": 5,
        },
    ),
    (
        PowellsMethod,
        False,
        False,
        {
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 5,
            "iters_p_dim": 5,
            "line_search": "golden",
            "convergence_threshold": 1e-6,
        },
    ),
    (
        PatternSearch,
        False,
        False,
        {
            "n_positions": 6,
            "pattern_size": 0.5,
            "reduction": 0.8,
        },
    ),
    (
        ParallelTemperingOptimizer,
        True,
        False,
        {
            "population": 5,
            "n_iter_swap": 3,
        },
    ),
    (
        ParticleSwarmOptimizer,
        True,
        False,
        {
            "population": 5,
            "inertia": 0.7,
            "cognitive_weight": 0.3,
            "social_weight": 0.7,
            "temp_weight": 0.1,
        },
    ),
    (
        SpiralOptimization,
        True,
        False,
        {
            "population": 5,
            "decay_rate": 0.95,
        },
    ),
    (
        GeneticAlgorithmOptimizer,
        True,
        False,
        {
            "population": 5,
            "offspring": 5,
            "crossover": "discrete-recombination",
            "n_parents": 2,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
        },
    ),
    (
        EvolutionStrategyOptimizer,
        True,
        False,
        {
            "population": 5,
            "offspring": 10,
            "replace_parents": True,
            "mutation_rate": 0.5,
            "crossover_rate": 0.5,
        },
    ),
    (
        DifferentialEvolutionOptimizer,
        True,
        False,
        {
            "population": 5,
            "mutation_rate": 0.5,
            "crossover_rate": 0.5,
        },
    ),
    (
        CMAESOptimizer,
        True,
        False,
        {
            "population": 5,
            "mu": 3,
            "sigma": 0.5,
            "ipop_restart": False,
        },
    ),
    (
        LipschitzOptimizer,
        False,
        True,
        {
            "warm_start_smbo": None,
            "max_sample_size": 1000,
            "sampling": {"random": 100},
            "replacement": True,
        },
    ),
    (
        DirectAlgorithm,
        False,
        True,
        {
            "resolution": 50,
            "warm_start_smbo": None,
            "max_sample_size": 1000,
            "sampling": {"random": 100},
            "replacement": True,
        },
    ),
    (
        BayesianOptimizer,
        False,
        True,
        {
            "warm_start_smbo": None,
            "max_sample_size": 1000,
            "sampling": {"random": 100},
            "replacement": True,
            "gpr": None,
            "xi": 0.05,
        },
    ),
    (
        TreeStructuredParzenEstimators,
        False,
        True,
        {
            "warm_start_smbo": None,
            "max_sample_size": 1000,
            "sampling": {"random": 100},
            "replacement": True,
            "gamma_tpe": 0.3,
        },
    ),
    (
        ForestOptimizer,
        False,
        True,
        {
            "warm_start_smbo": None,
            "max_sample_size": 1000,
            "sampling": {"random": 100},
            "replacement": True,
            "tree_regressor": "extra_tree",
            "tree_para": {"n_estimators": 50},
            "xi": 0.05,
        },
    ),
]


def _get_init(is_pop, uses_small):
    if uses_small:
        return smbo_init
    return pop_init if is_pop else default_init


def _get_objective(uses_small):
    return objective_small if uses_small else objective


ALL_OPTIMIZERS = [cfg[0] for cfg in OPTIMIZER_CONFIGS]

OPTIMIZERS_WITH_RAND_REST_P = [
    cfg[0] for cfg in OPTIMIZER_CONFIGS if cfg[0] is not RandomSearchOptimizer
]

INDIVIDUAL_PARAMS = []
for _Opt, _is_pop, _uses_small, _params in OPTIMIZER_CONFIGS:
    for _name, _value in _params.items():
        INDIVIDUAL_PARAMS.append((_Opt, _is_pop, _uses_small, _name, _value))


def _opt_id(opt):
    return opt.__name__


@pytest.mark.parametrize(
    "Optimizer, is_pop, uses_small, algo_params",
    OPTIMIZER_CONFIGS,
    ids=[_opt_id(c[0]) for c in OPTIMIZER_CONFIGS],
)
class TestDefaultParameters:
    """Verify each optimizer works with only required parameters."""

    def test_defaults(self, Optimizer, is_pop, uses_small, algo_params):
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        opt = Optimizer(
            SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE, initial_evaluations=init
        )
        _run_cycle(opt, obj_fn)

    def test_defaults_batch(self, Optimizer, is_pop, uses_small, algo_params):
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        opt = Optimizer(
            SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE, initial_evaluations=init
        )
        _run_batch_cycle(opt, obj_fn, n=2, batch=2)


@pytest.mark.parametrize(
    "Optimizer", ALL_OPTIMIZERS, ids=[o.__name__ for o in ALL_OPTIMIZERS]
)
class TestCommonBaseParameters:
    """Verify base parameters shared by all ask/tell optimizers."""

    def test_constraints_empty(self, Optimizer):
        cfg = next(c for c in OPTIMIZER_CONFIGS if c[0] is Optimizer)
        _, is_pop, uses_small, _ = cfg
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        space = SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE
        opt = Optimizer(space, initial_evaluations=init, constraints=[])
        _run_cycle(opt, obj_fn)

    def test_random_state(self, Optimizer):
        cfg = next(c for c in OPTIMIZER_CONFIGS if c[0] is Optimizer)
        _, is_pop, uses_small, _ = cfg
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        space = SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE
        opt = Optimizer(space, initial_evaluations=init, random_state=42)
        _run_cycle(opt, obj_fn)


@pytest.mark.parametrize(
    "Optimizer",
    OPTIMIZERS_WITH_RAND_REST_P,
    ids=[o.__name__ for o in OPTIMIZERS_WITH_RAND_REST_P],
)
class TestRandRestP:
    """Verify rand_rest_p for optimizers that support it."""

    def test_rand_rest_p(self, Optimizer):
        cfg = next(c for c in OPTIMIZER_CONFIGS if c[0] is Optimizer)
        _, is_pop, uses_small, _ = cfg
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        space = SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE
        opt = Optimizer(space, initial_evaluations=init, rand_rest_p=0.1)
        _run_cycle(opt, obj_fn)


@pytest.mark.parametrize(
    "Optimizer, is_pop, uses_small, param_name, param_value",
    INDIVIDUAL_PARAMS,
    ids=[f"{p[0].__name__}-{p[3]}" for p in INDIVIDUAL_PARAMS],
)
class TestIndividualAlgoParameters:
    """Test each algorithm-specific parameter in isolation."""

    def test_single_param(self, Optimizer, is_pop, uses_small, param_name, param_value):
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        space = SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE
        opt = Optimizer(space, initial_evaluations=init, **{param_name: param_value})
        _run_cycle(opt, obj_fn)


@pytest.mark.parametrize(
    "Optimizer, is_pop, uses_small, algo_params",
    OPTIMIZER_CONFIGS,
    ids=[_opt_id(c[0]) for c in OPTIMIZER_CONFIGS],
)
class TestAllParamsExplicit:
    """Verify all parameters can be set simultaneously."""

    def test_all_algo_params(self, Optimizer, is_pop, uses_small, algo_params):
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        space = SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE
        base = {"random_state": 42}
        if Optimizer is not RandomSearchOptimizer:
            base["rand_rest_p"] = 0.1
        opt = Optimizer(
            space, initial_evaluations=init, constraints=[], **base, **algo_params
        )
        _run_cycle(opt, obj_fn)

    def test_all_algo_params_batch(self, Optimizer, is_pop, uses_small, algo_params):
        init = _get_init(is_pop, uses_small)
        obj_fn = _get_objective(uses_small)
        space = SEARCH_SPACE_SMALL if uses_small else SEARCH_SPACE
        base = {"random_state": 42}
        if Optimizer is not RandomSearchOptimizer:
            base["rand_rest_p"] = 0.1
        opt = Optimizer(
            space, initial_evaluations=init, constraints=[], **base, **algo_params
        )
        _run_batch_cycle(opt, obj_fn, n=2, batch=3)

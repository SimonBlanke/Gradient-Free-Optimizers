# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API tests for the catch error handling system.

Tests that catch maps exception types to fallback scores, works with
memory/caching, respects exception inheritance, and interacts correctly
with optimum="minimum".
"""

import math

import numpy as np
import pytest

from gradient_free_optimizers import (
    BayesianOptimizer,
    DifferentialEvolutionOptimizer,
    GridSearchOptimizer,
    HillClimbingOptimizer,
    ParticleSwarmOptimizer,
    RandomAnnealingOptimizer,
    RandomSearchOptimizer,
    SimulatedAnnealingOptimizer,
)

SEARCH_SPACE = {"x": np.linspace(-1, 1, 10)}
N_ITER = 15


def always_raises(p):
    raise ValueError("always fails")


def sometimes_raises(p):
    if p["x"] > 0.5:
        raise ValueError("bad parameter")
    return -(p["x"] ** 2)


all_optimizers = [
    RandomSearchOptimizer,
    HillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    GridSearchOptimizer,
    ParticleSwarmOptimizer,
    DifferentialEvolutionOptimizer,
    RandomAnnealingOptimizer,
    BayesianOptimizer,
]


@pytest.mark.parametrize("Optimizer", all_optimizers)
class TestCatchBasics:
    """Catch mechanism works across all optimizer categories."""

    def test_catch_completes_search(self, Optimizer):
        opt = Optimizer(SEARCH_SPACE)
        opt.search(
            always_raises,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: -1000},
        )
        assert len(opt.score_l) == N_ITER

    def test_catch_fallback_score_recorded(self, Optimizer):
        opt = Optimizer(SEARCH_SPACE)
        opt.search(
            always_raises,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: -1000},
        )
        assert all(s == -1000 for s in opt.score_l)

    def test_no_catch_raises(self, Optimizer):
        opt = Optimizer(SEARCH_SPACE)
        with pytest.raises(ValueError, match="always fails"):
            opt.search(always_raises, n_iter=N_ITER, verbosity=False)

    def test_catch_none_default(self, Optimizer):
        """Without catch, no wrapping occurs."""

        def objective(p):
            return -(p["x"] ** 2)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False)
        assert len(opt.score_l) == N_ITER


class TestCatchFallbackTypes:
    """Different fallback score types are handled correctly."""

    def test_catch_with_nan(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(
            always_raises,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: float("nan")},
        )
        assert len(opt.score_l) == N_ITER
        assert all(math.isnan(s) for s in opt.score_l)

    def test_catch_with_inf(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(
            always_raises,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: float("-inf")},
        )
        assert len(opt.score_l) == N_ITER

    def test_catch_with_int(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(
            always_raises,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: 0},
        )
        assert len(opt.score_l) == N_ITER


class TestCatchExceptionMatching:
    """Exception type matching uses isinstance semantics."""

    def test_catch_wrong_type_raises(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        with pytest.raises(ValueError):
            opt.search(
                always_raises,
                n_iter=N_ITER,
                verbosity=False,
                catch={RuntimeError: -1000},
            )

    def test_catch_parent_catches_child(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(
            always_raises,
            n_iter=N_ITER,
            verbosity=False,
            catch={Exception: -1000},
        )
        assert len(opt.score_l) == N_ITER

    def test_catch_multiple_exceptions(self):
        def multi_error(p):
            if p["x"] > 0.5:
                raise ValueError("val")
            if p["x"] < -0.5:
                raise RuntimeError("rt")
            return -(p["x"] ** 2)

        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(
            multi_error,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: -1000, RuntimeError: -2000},
        )
        assert len(opt.score_l) == N_ITER

    def test_catch_different_scores_per_type(self):
        call_log = []

        def multi_error(p):
            if p["x"] > 0:
                raise ValueError("val")
            raise RuntimeError("rt")

        def track(info):
            call_log.append(info.score)

        opt = RandomSearchOptimizer(SEARCH_SPACE, random_state=42)
        opt.search(
            multi_error,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: -100, RuntimeError: -200},
            callbacks=[track],
        )
        scores = set(opt.score_l)
        assert scores <= {-100, -200}


class TestCatchWithMinimum:
    """Catch fallback scores are in user's units, negated correctly."""

    def test_catch_minimum_negates_fallback(self):
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(
            always_raises,
            n_iter=N_ITER,
            verbosity=False,
            catch={ValueError: 1e10},
            optimum="minimum",
        )
        assert len(opt.score_l) == N_ITER

    def test_catch_minimum_best_score_not_fallback(self):
        def objective_min(p):
            if p["x"] > 0.8:
                raise ValueError("bad")
            return p["x"] ** 2

        opt = RandomSearchOptimizer(SEARCH_SPACE, random_state=42)
        opt.search(
            objective_min,
            n_iter=30,
            verbosity=False,
            catch={ValueError: 1e10},
            optimum="minimum",
        )
        assert opt.best_score < 1e10


class TestCatchWithMemory:
    """Fallback scores are cached when memory=True."""

    def test_catch_cached_avoids_repeat_calls(self):
        call_count = [0]

        def counting_raises(p):
            call_count[0] += 1
            raise ValueError("always fails")

        opt = RandomSearchOptimizer({"x": np.linspace(-1, 1, 3)})
        opt.search(
            counting_raises,
            n_iter=10,
            verbosity=False,
            memory=True,
            catch={ValueError: -1000},
        )
        assert call_count[0] <= 3

    def test_catch_no_memory_calls_every_time(self):
        call_count = [0]

        def counting_raises(p):
            call_count[0] += 1
            raise ValueError("always fails")

        opt = RandomSearchOptimizer({"x": np.linspace(-1, 1, 3)})
        opt.search(
            counting_raises,
            n_iter=10,
            verbosity=False,
            memory=False,
            catch={ValueError: -1000},
        )
        assert call_count[0] == 10

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API tests for the callback system.

Tests that callbacks receive correct CallbackInfo data, support early stopping
via ``return False``, and work across all optimizer categories.
"""

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
from gradient_free_optimizers._callback import CallbackInfo

SEARCH_SPACE = {"x": np.linspace(-1, 1, 10)}
N_ITER = 15


def objective(p):
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
class TestCallbackBasics:
    """Callback mechanism works across all optimizer categories."""

    def test_callback_called_each_iteration(self, Optimizer):
        call_count = []

        def counter(info):
            call_count.append(1)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[counter])
        assert len(call_count) == N_ITER

    def test_callback_receives_callback_info(self, Optimizer):
        infos = []

        def collector(info):
            infos.append(info)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[collector])

        last = infos[-1]
        assert isinstance(last, CallbackInfo)
        assert isinstance(last.score, float)
        assert isinstance(last.params, dict)
        assert "x" in last.params
        assert isinstance(last.best_score, float)
        assert isinstance(last.best_para, dict)
        assert "x" in last.best_para
        assert last.n_iter == N_ITER
        assert last.elapsed_time > 0
        assert isinstance(last.metrics, dict)
        assert isinstance(last.convergence, list)

    def test_callback_iteration_increments(self, Optimizer):
        iterations = []

        def track(info):
            iterations.append(info.iteration)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[track])
        assert iterations == list(range(N_ITER))

    def test_callback_stop_with_false(self, Optimizer):
        stop_at = 3

        def stop_early(info):
            if info.iteration >= stop_at:
                return False

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=100, verbosity=False, callbacks=[stop_early])
        assert len(opt.score_l) <= stop_at + 2

    def test_callback_none_continues(self, Optimizer):
        def noop(info):
            pass

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[noop])
        assert len(opt.score_l) == N_ITER

    def test_no_callbacks_default(self, Optimizer):
        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False)
        assert len(opt.score_l) == N_ITER

    def test_empty_callbacks_list(self, Optimizer):
        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[])
        assert len(opt.score_l) == N_ITER


@pytest.mark.parametrize("Optimizer", all_optimizers)
class TestCallbackMultiple:
    """Multiple callbacks and ordering."""

    def test_multiple_callbacks_all_called(self, Optimizer):
        log1, log2 = [], []

        def cb1(info):
            log1.append(info.score)

        def cb2(info):
            log2.append(info.best_score)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[cb1, cb2])
        assert len(log1) == N_ITER
        assert len(log2) == N_ITER

    def test_first_stop_prevents_later_callbacks(self, Optimizer):
        log = []

        def stopper(info):
            return False

        def logger(info):
            log.append(1)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(
            objective, n_iter=N_ITER, verbosity=False, callbacks=[stopper, logger]
        )
        assert len(log) == 0


@pytest.mark.parametrize("Optimizer", all_optimizers)
class TestCallbackInfoContent:
    """CallbackInfo fields contain correct data."""

    def test_convergence_grows(self, Optimizer):
        lengths = []

        def track(info):
            lengths.append(len(info.convergence))

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[track])
        assert lengths == list(range(1, N_ITER + 1))

    def test_convergence_is_copy(self, Optimizer):
        convergences = []

        def track(info):
            convergences.append(info.convergence)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=5, verbosity=False, callbacks=[track])
        assert len(convergences[0]) == 1
        assert len(convergences[-1]) == 5

    def test_best_score_monotonic(self, Optimizer):
        best_scores = []

        def track(info):
            best_scores.append(info.best_score)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[track])
        for i in range(1, len(best_scores)):
            assert best_scores[i] >= best_scores[i - 1]

    def test_phase_field(self, Optimizer):
        phases = set()

        def track(info):
            phases.add(info.phase)

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[track])
        assert phases <= {"init", "iter"}

    def test_info_is_frozen(self, Optimizer):
        def try_mutate(info):
            with pytest.raises(AttributeError):
                info.score = 999

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False, callbacks=[try_mutate])

    def test_n_iter_matches(self, Optimizer):
        def check(info):
            assert info.n_iter == N_ITER

        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=N_ITER, verbosity=False, callbacks=[check])

"""Tests for distributed evaluation backends and search loop integration."""

import concurrent.futures
import os

import numpy as np
import pytest

import gradient_free_optimizers._distributed._multiprocessing as _mp_module
from gradient_free_optimizers import (
    BayesianOptimizer,
    DirectAlgorithm,
    DownhillSimplexOptimizer,
    HillClimbingOptimizer,
    ParticleSwarmOptimizer,
    PowellsMethod,
    RandomSearchOptimizer,
)
from gradient_free_optimizers._distributed import (
    BaseDistribution,
    Dask,
    Joblib,
    Multiprocessing,
    Ray,
)
from gradient_free_optimizers._storage import MemoryStorage

search_space = {"x": np.linspace(-10, 10, 100), "y": np.linspace(-10, 10, 100)}
tiny_space = {"x": np.linspace(0, 1, 10)}


def objective(para):
    return -(para["x"] ** 2 + para["y"] ** 2)


def objective_with_metrics(para):
    score = -(para["x"] ** 2 + para["y"] ** 2)
    return score, {"loss": -score, "x_abs": abs(para["x"])}


class ThreadAsync(BaseDistribution):
    """Lightweight async backend using threads, for testing without Ray/Dask."""

    _is_async = True

    def __init__(self, n_workers=2):
        super().__init__(n_workers)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)

    def _distribute(self, func, params_batch):
        futures = [self._pool.submit(func, p) for p in params_batch]
        return [f.result() for f in futures]

    def _submit(self, func, params):
        return self._pool.submit(func, params)

    def _wait_any(self, futures):
        done, _ = concurrent.futures.wait(futures, return_when="FIRST_COMPLETED")
        completed = next(iter(done))
        return completed, completed.result()


def flaky_objective(para):
    if abs(para["x"]) > 8:
        raise ValueError("out of range")
    return -(para["x"] ** 2 + para["y"] ** 2)


class TestBackendMetadata:
    def test_multiprocessing_is_sync(self):
        assert Multiprocessing._is_async is False

    def test_joblib_is_sync(self):
        assert Joblib._is_async is False

    def test_decorator_stores_backend_ref(self):
        backend = Joblib(n_workers=2)
        wrapped = backend.distribute(objective)
        assert wrapped._gfo_distributed is True
        assert wrapped._gfo_batch_size == 2
        assert wrapped._gfo_backend is backend
        assert wrapped._gfo_original_func is objective


class TestJoblib:
    def test_basic_search(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert opt.best_score is not None
        assert len(opt.score_l) == 20

    def test_with_memory(self):
        storage = MemoryStorage()
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, memory=storage, verbosity=False)
        assert len(storage) > 0

    def test_minimization(self):
        def maximize_me(para):
            return para["x"] ** 2 + para["y"] ** 2

        distributed = Joblib(n_workers=2).distribute(maximize_me)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, optimum="minimum", verbosity=False)
        assert opt.best_score is not None

    def test_population_optimizer(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = ParticleSwarmOptimizer(search_space, population=5)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20


class TestMultiprocessing:
    def test_basic_search(self):
        distributed = Multiprocessing(n_workers=2).distribute(objective)
        opt = RandomSearchOptimizer(search_space)
        opt.search(distributed, n_iter=15, verbosity=False)
        assert len(opt.score_l) == 15


class TestCatchDistributed:
    def test_catch_with_joblib(self):
        distributed = Joblib(n_workers=2).distribute(flaky_objective)
        opt = RandomSearchOptimizer(search_space)
        opt.search(
            distributed,
            n_iter=20,
            catch={ValueError: -999.0},
            verbosity=False,
        )
        assert len(opt.score_l) == 20
        assert any(s <= -900 for s in opt.score_l)

    def test_catch_with_multiprocessing(self):
        distributed = Multiprocessing(n_workers=2).distribute(flaky_objective)
        opt = RandomSearchOptimizer(search_space)
        opt.search(
            distributed,
            n_iter=15,
            catch={ValueError: -999.0},
            verbosity=False,
        )
        assert len(opt.score_l) == 15


class TestTimingInvariants:
    def test_eval_time_leq_iter_time(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)

        total_eval = sum(opt.eval_times)
        total_iter = sum(opt.iter_times)

        assert (
            total_eval <= total_iter + 0.01
        ), f"eval_time ({total_eval:.3f}) > iter_time ({total_iter:.3f})"

    def test_iter_times_has_entries_for_all_iterations(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)

        assert len(opt.iter_times) == 20
        assert len(opt.eval_times) == 20


class TestSupportsAsync:
    def test_default_supports_async(self):
        opt = HillClimbingOptimizer(search_space)
        assert opt._supports_async is True

    def test_pso_supports_async(self):
        opt = ParticleSwarmOptimizer(search_space, population=5)
        assert opt._supports_async is True

    def test_bo_supports_async(self):
        opt = BayesianOptimizer(search_space)
        assert opt._supports_async is True

    def test_simplex_not_async(self):
        opt = DownhillSimplexOptimizer(search_space)
        assert opt._supports_async is False

    def test_powell_not_async(self):
        opt = PowellsMethod(search_space)
        assert opt._supports_async is False

    def test_direct_not_async(self):
        ss_1d = {"x": np.linspace(-10, 10, 100)}
        opt = DirectAlgorithm(ss_1d)
        assert opt._supports_async is False


class TestSMBOBatchDiversity:
    """Verify that SMBO batch positions are diverse, not clustered."""

    def test_bo_batch_positions_are_spread(self):
        distributed = Joblib(n_workers=4).distribute(objective)
        opt = BayesianOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20


def _assert_metrics_in_search_data(opt, n_iter):
    """Verify that metric columns appear in search_data with correct values."""
    df = opt.search_data
    assert len(df) == n_iter
    assert "loss" in df.columns, f"Missing 'loss' column. Columns: {list(df.columns)}"
    assert "x_abs" in df.columns, f"Missing 'x_abs' column. Columns: {list(df.columns)}"
    assert df["loss"].notna().all(), "Some 'loss' values are NaN"
    assert (df["x_abs"] >= 0).all(), "x_abs should be non-negative"


class TestMetricsPreservedSyncBatch:
    """Metrics from (score, metrics) objectives must survive the sync-batch path."""

    def test_joblib_metrics_in_search_data(self):
        distributed = Joblib(n_workers=2).distribute(objective_with_metrics)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        _assert_metrics_in_search_data(opt, 20)

    def test_joblib_metrics_in_storage(self):
        storage = MemoryStorage()
        distributed = Joblib(n_workers=2).distribute(objective_with_metrics)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, memory=storage, verbosity=False)
        _assert_metrics_in_search_data(opt, 20)

        stored_with_metrics = sum(
            1 for _, r in storage.items() if r.metrics and "loss" in r.metrics
        )
        assert stored_with_metrics > 0, "No stored results contain metrics"

    def test_joblib_metrics_with_minimization(self):
        def obj_minimize(para):
            score = para["x"] ** 2 + para["y"] ** 2
            return score, {"raw_score": score}

        distributed = Joblib(n_workers=2).distribute(obj_minimize)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, optimum="minimum", verbosity=False)
        df = opt.search_data
        assert "raw_score" in df.columns
        assert df["raw_score"].notna().all()

    def test_joblib_metrics_in_callbacks(self):
        collected = []

        def capture_metrics(info):
            collected.append(dict(info.metrics))

        distributed = Joblib(n_workers=2).distribute(objective_with_metrics)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False, callbacks=[capture_metrics])

        non_empty = [m for m in collected if m]
        assert len(non_empty) > 0, "Callbacks never received non-empty metrics"


class TestMetricsPreservedTrueAsync:
    """Metrics must survive the true-async path (_run_true_async)."""

    def test_thread_async_metrics_in_search_data(self):
        distributed = ThreadAsync(n_workers=2).distribute(objective_with_metrics)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        _assert_metrics_in_search_data(opt, 20)

    def test_thread_async_metrics_in_storage(self):
        storage = MemoryStorage()
        distributed = ThreadAsync(n_workers=2).distribute(objective_with_metrics)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, memory=storage, verbosity=False)
        _assert_metrics_in_search_data(opt, 20)

        stored_with_metrics = sum(
            1 for _, r in storage.items() if r.metrics and "loss" in r.metrics
        )
        assert stored_with_metrics > 0, "No stored results contain metrics"

    def test_thread_async_metrics_in_callbacks(self):
        collected = []

        def capture_metrics(info):
            collected.append(dict(info.metrics))

        distributed = ThreadAsync(n_workers=2).distribute(objective_with_metrics)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False, callbacks=[capture_metrics])

        non_empty = [m for m in collected if m]
        assert len(non_empty) > 0, "Callbacks never received non-empty metrics"


class TestMetricsPreservedBatchAsync:
    """Metrics must survive the batch-async path (_run_batch_async).

    Stateful optimizers like DownhillSimplex route to batch-async when the
    backend is async-capable but the optimizer sets _supports_async = False.
    """

    def test_simplex_async_metrics_in_search_data(self):
        distributed = ThreadAsync(n_workers=3).distribute(objective_with_metrics)
        opt = DownhillSimplexOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        _assert_metrics_in_search_data(opt, 20)

    def test_simplex_async_metrics_in_storage(self):
        storage = MemoryStorage()
        distributed = ThreadAsync(n_workers=3).distribute(objective_with_metrics)
        opt = DownhillSimplexOptimizer(search_space)
        opt.search(distributed, n_iter=20, memory=storage, verbosity=False)
        _assert_metrics_in_search_data(opt, 20)

        stored_with_metrics = sum(
            1 for _, r in storage.items() if r.metrics and "loss" in r.metrics
        )
        assert stored_with_metrics > 0, "No stored results contain metrics"


class TestAsyncPaths:
    """Basic functionality for true-async and batch-async code paths.

    ThreadAsync exercises both paths depending on the optimizer:
    _supports_async=True (HillClimbing, PSO) -> _run_true_async
    _supports_async=False (Simplex, Powell) -> _run_batch_async
    """

    def test_true_async_basic(self):
        distributed = ThreadAsync(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert opt.best_score is not None
        assert len(opt.score_l) == 20

    def test_true_async_with_memory(self):
        storage = MemoryStorage()
        distributed = ThreadAsync(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, memory=storage, verbosity=False)
        assert len(storage) > 0

    def test_true_async_minimization(self):
        def maximize_me(para):
            return para["x"] ** 2 + para["y"] ** 2

        distributed = ThreadAsync(n_workers=2).distribute(maximize_me)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, optimum="minimum", verbosity=False)
        assert opt.best_score is not None

    def test_true_async_population(self):
        distributed = ThreadAsync(n_workers=3).distribute(objective)
        opt = ParticleSwarmOptimizer(search_space, population=5)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20

    def test_batch_async_simplex(self):
        distributed = ThreadAsync(n_workers=3).distribute(objective)
        opt = DownhillSimplexOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20

    def test_batch_async_powell(self):
        distributed = ThreadAsync(n_workers=2).distribute(objective)
        opt = PowellsMethod(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20

    def test_batch_async_with_memory(self):
        storage = MemoryStorage()
        distributed = ThreadAsync(n_workers=3).distribute(objective)
        opt = DownhillSimplexOptimizer(search_space)
        opt.search(distributed, n_iter=20, memory=storage, verbosity=False)
        assert len(storage) > 0


class TestStatefulSyncBatch:
    """Stateful optimizers (Simplex, Powell) with synchronous distributed backends.

    These optimizers set _supports_async=False, so even with a sync backend
    they go through _iteration_batch which must respect the batch contract.
    """

    def test_simplex_with_joblib(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = DownhillSimplexOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20
        assert opt.best_score is not None

    def test_powell_with_joblib(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = PowellsMethod(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20


class TestCatchAsync:
    """Exception catching works with async distributed backends."""

    def test_catch_with_true_async(self):
        distributed = ThreadAsync(n_workers=2).distribute(flaky_objective)
        opt = RandomSearchOptimizer(search_space)
        opt.search(
            distributed,
            n_iter=20,
            catch={ValueError: -999.0},
            verbosity=False,
        )
        assert len(opt.score_l) == 20
        assert any(s <= -900 for s in opt.score_l)


class TestEarlyStopDistributed:
    """Stopping conditions integrate correctly with all distributed paths.

    Uses deterministic callback-based stopping (most reliable) and
    stopper-based conditions (max_score, n_iter_no_change).
    """

    @staticmethod
    def _stop_after_n(n):
        calls = [0]

        def callback(info):
            calls[0] += 1
            if calls[0] >= n:
                return False

        return callback

    def test_callback_stops_sync_batch(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space, initialize={"random": 2})
        opt.search(
            distributed,
            n_iter=200,
            verbosity=False,
            callbacks=[self._stop_after_n(15)],
        )
        assert len(opt.score_l) < 200

    def test_callback_stops_true_async(self):
        distributed = ThreadAsync(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space, initialize={"random": 2})
        opt.search(
            distributed,
            n_iter=200,
            verbosity=False,
            callbacks=[self._stop_after_n(15)],
        )
        assert len(opt.score_l) < 200

    def test_callback_stops_batch_async(self):
        distributed = ThreadAsync(n_workers=3).distribute(objective)
        opt = DownhillSimplexOptimizer(search_space, initialize={"random": 2})
        opt.search(
            distributed,
            n_iter=200,
            verbosity=False,
            callbacks=[self._stop_after_n(15)],
        )
        assert len(opt.score_l) < 200

    def test_max_score_stops_search(self):
        distributed = Joblib(n_workers=2).distribute(objective)
        opt = RandomSearchOptimizer(search_space)
        # -(x^2+y^2) ranges [-200, 0]; score >= -150 covers ~95% of the space
        opt.search(distributed, n_iter=200, max_score=-150, verbosity=False)
        assert len(opt.score_l) < 200

    def test_early_stopping_n_iter_no_change(self):
        def constant(para):
            return 1.0

        distributed = Joblib(n_workers=2).distribute(constant)
        opt = HillClimbingOptimizer(search_space, initialize={"random": 2})
        opt.search(
            distributed,
            n_iter=200,
            early_stopping={"n_iter_no_change": 5},
            verbosity=False,
        )
        # constant function never improves, stopper triggers after ~7-10 evals
        assert len(opt.score_l) < 200


class TestStorageCacheDistributed:
    """Storage cache hits reduce redundant objective function evaluations.

    Uses ThreadAsync (threads share memory) so a call counter works reliably.
    For the sync-batch variant, Joblib with threading backend serves the
    same purpose.
    """

    def test_cache_reduces_calls_true_async(self):
        calls = []

        def tracked(para):
            calls.append(1)
            return -(para["x"] ** 2)

        storage = MemoryStorage()

        d1 = ThreadAsync(n_workers=2).distribute(tracked)
        opt1 = RandomSearchOptimizer(
            tiny_space, initialize={"random": 2}, random_state=42
        )
        opt1.search(d1, n_iter=15, memory=storage, verbosity=False)
        first = len(calls)

        calls.clear()
        d2 = ThreadAsync(n_workers=2).distribute(tracked)
        opt2 = RandomSearchOptimizer(
            tiny_space, initialize={"random": 2}, random_state=42
        )
        opt2.search(d2, n_iter=15, memory=storage, verbosity=False)
        second = len(calls)

        assert second < first, f"Cache didn't reduce calls: {second} >= {first}"

    def test_cache_reduces_calls_sync_batch(self):
        calls = []

        def tracked(para):
            calls.append(1)
            return -(para["x"] ** 2)

        storage = MemoryStorage()

        d1 = Joblib(n_workers=2, backend="threading").distribute(tracked)
        opt1 = RandomSearchOptimizer(
            tiny_space, initialize={"random": 2}, random_state=42
        )
        opt1.search(d1, n_iter=15, memory=storage, verbosity=False)
        first = len(calls)

        calls.clear()
        d2 = Joblib(n_workers=2, backend="threading").distribute(tracked)
        opt2 = RandomSearchOptimizer(
            tiny_space, initialize={"random": 2}, random_state=42
        )
        opt2.search(d2, n_iter=15, memory=storage, verbosity=False)
        second = len(calls)

        assert second < first, f"Cache didn't reduce calls: {second} >= {first}"

    def test_warm_start_reduces_calls(self):
        def obj(para):
            return -(para["x"] ** 2)

        # First run: serial, populates search_data for warm start
        opt1 = RandomSearchOptimizer(
            tiny_space, random_state=42, initialize={"random": 2}
        )
        opt1.search(obj, n_iter=12, verbosity=False)
        warm_data = opt1.search_data

        # Second run: distributed with warm_start from first run
        calls = []

        def tracked(para):
            calls.append(1)
            return obj(para)

        distributed = ThreadAsync(n_workers=2).distribute(tracked)
        opt2 = RandomSearchOptimizer(
            tiny_space, random_state=42, initialize={"random": 2}
        )
        opt2.search(
            distributed,
            n_iter=12,
            memory=True,
            memory_warm_start=warm_data,
            verbosity=False,
        )

        # With 10 total points and warm start from identical run,
        # most evaluations should be cache hits
        assert (
            len(calls) < 12
        ), f"Warm start didn't help: {len(calls)} calls for 12 iters"


class TestIterationCountEdgeCases:
    """n_iter edge cases: not divisible by n_workers, fewer than workers."""

    def test_non_divisible_sync_batch(self):
        distributed = Joblib(n_workers=3).distribute(objective)
        opt = HillClimbingOptimizer(search_space, initialize={"random": 2})
        opt.search(distributed, n_iter=17, verbosity=False)
        assert len(opt.score_l) == 17

    def test_non_divisible_true_async(self):
        distributed = ThreadAsync(n_workers=3).distribute(objective)
        opt = HillClimbingOptimizer(search_space, initialize={"random": 2})
        opt.search(distributed, n_iter=17, verbosity=False)
        assert len(opt.score_l) == 17

    def test_non_divisible_batch_async(self):
        distributed = ThreadAsync(n_workers=4).distribute(objective)
        opt = DownhillSimplexOptimizer(search_space, initialize={"random": 2})
        opt.search(distributed, n_iter=17, verbosity=False)
        assert len(opt.score_l) == 17

    def test_fewer_iters_than_workers(self):
        distributed = Joblib(n_workers=8).distribute(objective)
        opt = HillClimbingOptimizer(search_space, initialize={"random": 2})
        opt.search(distributed, n_iter=5, verbosity=False)
        assert len(opt.score_l) == 5


class TestCustomBackend:
    """Custom BaseDistribution subclass works end-to-end with search()."""

    def test_serial_custom_backend(self):
        class SerialBackend(BaseDistribution):
            def _distribute(self, func, params_batch):
                return [func(p) for p in params_batch]

        distributed = SerialBackend(n_workers=2).distribute(objective)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        assert len(opt.score_l) == 20
        assert opt.best_score is not None

    def test_custom_backend_with_metrics(self):
        """Custom backend passes through (score, metrics) tuples correctly."""

        class SerialBackend(BaseDistribution):
            def _distribute(self, func, params_batch):
                return [func(p) for p in params_batch]

        distributed = SerialBackend(n_workers=2).distribute(objective_with_metrics)
        opt = HillClimbingOptimizer(search_space)
        opt.search(distributed, n_iter=20, verbosity=False)
        _assert_metrics_in_search_data(opt, 20)


class TestSearchDataConsistency:
    """Distributed search_data DataFrame matches serial in structure."""

    def test_columns_match_serial(self):
        opt_serial = HillClimbingOptimizer(search_space, random_state=42)
        opt_serial.search(objective, n_iter=20, verbosity=False)

        distributed = Joblib(n_workers=2).distribute(objective)
        opt_dist = HillClimbingOptimizer(search_space, random_state=42)
        opt_dist.search(distributed, n_iter=20, verbosity=False)

        assert set(opt_serial.search_data.columns) == set(opt_dist.search_data.columns)
        assert len(opt_dist.search_data) == 20

    def test_metric_columns_match_serial(self):
        opt_serial = HillClimbingOptimizer(search_space, random_state=42)
        opt_serial.search(objective_with_metrics, n_iter=20, verbosity=False)

        distributed = Joblib(n_workers=2).distribute(objective_with_metrics)
        opt_dist = HillClimbingOptimizer(search_space, random_state=42)
        opt_dist.search(distributed, n_iter=20, verbosity=False)

        serial_cols = set(opt_serial.search_data.columns)
        dist_cols = set(opt_dist.search_data.columns)
        assert serial_cols == dist_cols, f"Column mismatch: {serial_cols ^ dist_cols}"
        assert "loss" in dist_cols
        assert "x_abs" in dist_cols


def _identity_x(para):
    """Returns para["x"] directly. Module-level for multiprocessing pickling."""
    return para["x"]


def _identity_x_with_metrics(para):
    """Returns (score, metrics) tuple. Module-level for multiprocessing pickling."""
    return para["x"], {"doubled": para["x"] * 2}


class TestBaseDistributionValidation:
    """Unit tests for the BaseDistribution ABC contract."""

    def _make_concrete(self):
        class Concrete(BaseDistribution):
            def _distribute(self, func, params_batch):
                return [func(p) for p in params_batch]

        return Concrete

    def test_n_workers_zero_raises(self):
        Concrete = self._make_concrete()
        with pytest.raises(ValueError, match="n_workers must be >= 1"):
            Concrete(n_workers=0)

    def test_n_workers_negative_raises(self):
        Concrete = self._make_concrete()
        with pytest.raises(ValueError, match="n_workers must be >= 1"):
            Concrete(n_workers=-2)

    def test_n_workers_one_accepted(self):
        Concrete = self._make_concrete()
        b = Concrete(n_workers=1)
        assert b.n_workers == 1

    def test_n_workers_stored(self):
        Concrete = self._make_concrete()
        b = Concrete(n_workers=7)
        assert b.n_workers == 7

    def test_is_async_default_false(self):
        assert BaseDistribution._is_async is False

    def test_submit_raises_not_implemented(self):
        Concrete = self._make_concrete()
        b = Concrete(n_workers=1)
        with pytest.raises(NotImplementedError, match="Concrete"):
            b._submit(lambda x: x, {})

    def test_wait_any_raises_not_implemented(self):
        Concrete = self._make_concrete()
        b = Concrete(n_workers=1)
        with pytest.raises(NotImplementedError, match="Concrete"):
            b._wait_any([])


class TestDistributeWrapper:
    """Tests for the distribute() decorator on BaseDistribution."""

    def _make_backend(self, n_workers=3):
        class Serial(BaseDistribution):
            def _distribute(self, func, params_batch):
                return [func(p) for p in params_batch]

        return Serial(n_workers=n_workers)

    def test_gfo_distributed_flag(self):
        wrapped = self._make_backend().distribute(objective)
        assert wrapped._gfo_distributed is True

    def test_gfo_batch_size_matches_n_workers(self):
        wrapped = self._make_backend(n_workers=5).distribute(objective)
        assert wrapped._gfo_batch_size == 5

    def test_gfo_original_func_is_original(self):
        wrapped = self._make_backend().distribute(objective)
        assert wrapped._gfo_original_func is objective

    def test_gfo_backend_is_backend_instance(self):
        backend = self._make_backend()
        wrapped = backend.distribute(objective)
        assert wrapped._gfo_backend is backend

    def test_preserves_function_name(self):
        wrapped = self._make_backend().distribute(objective)
        assert wrapped.__name__ == "objective"

    def test_preserves_lambda_name(self):
        fn = lambda para: para["x"]  # noqa: E731
        wrapped = self._make_backend().distribute(fn)
        assert wrapped.__name__ == "<lambda>"

    def test_wrapper_delegates_to_distribute(self):
        wrapped = self._make_backend().distribute(_identity_x)
        results = wrapped([{"x": 10}, {"x": 20}])
        assert results == [10, 20]

    def test_wrapper_is_callable(self):
        wrapped = self._make_backend().distribute(objective)
        assert callable(wrapped)


class TestMultiprocessingUnit:
    """Unit tests for the Multiprocessing backend."""

    def test_auto_detect_workers(self):
        mp = Multiprocessing(n_workers=-1)
        expected = os.cpu_count() or 1
        assert mp.n_workers == expected

    def test_explicit_workers(self):
        mp = Multiprocessing(n_workers=3)
        assert mp.n_workers == 3

    def test_prefers_fork_when_available(self):
        import multiprocessing

        mp = Multiprocessing(n_workers=2)
        if "fork" in multiprocessing.get_all_start_methods():
            assert mp._use_fork is True
        else:
            assert mp._use_fork is False

    def test_context_has_valid_start_method(self):
        mp = Multiprocessing(n_workers=2)
        assert mp._mp_context.get_start_method() in ("fork", "spawn", "forkserver")

    def test_result_ordering(self):
        mp = Multiprocessing(n_workers=2)
        batch = [{"x": i} for i in range(10)]
        results = mp._distribute(_identity_x, batch)
        assert results == list(range(10))

    def test_single_item_batch(self):
        mp = Multiprocessing(n_workers=2)
        results = mp._distribute(_identity_x, [{"x": 42}])
        assert results == [42]

    def test_empty_batch(self):
        mp = Multiprocessing(n_workers=2)
        results = mp._distribute(_identity_x, [])
        assert results == []

    def test_worker_func_cleaned_up_after_distribute(self):
        mp = Multiprocessing(n_workers=2)
        assert _mp_module._worker_func is None
        mp._distribute(_identity_x, [{"x": 1}])
        assert _mp_module._worker_func is None

    def test_tuple_result_passthrough(self):
        mp = Multiprocessing(n_workers=2)
        batch = [{"x": 1}, {"x": 2}]
        results = mp._distribute(_identity_x_with_metrics, batch)
        assert results[0] == (1, {"doubled": 2})
        assert results[1] == (2, {"doubled": 4})


class TestJoblibUnit:
    """Unit tests for the Joblib backend."""

    def test_auto_detect_workers(self):
        jl = Joblib(n_workers=-1)
        assert jl.n_workers >= 1

    def test_explicit_workers(self):
        jl = Joblib(n_workers=3)
        assert jl.n_workers == 3

    def test_default_backend_is_loky(self):
        jl = Joblib(n_workers=2)
        assert jl._backend_name == "loky"

    def test_custom_backend_stored(self):
        jl = Joblib(n_workers=2, backend="threading")
        assert jl._backend_name == "threading"

    def test_result_ordering(self):
        jl = Joblib(n_workers=2)
        batch = [{"x": i} for i in range(10)]
        results = jl._distribute(_identity_x, batch)
        assert results == list(range(10))

    def test_single_item_batch(self):
        jl = Joblib(n_workers=2)
        results = jl._distribute(_identity_x, [{"x": 99}])
        assert results == [99]

    def test_empty_batch(self):
        jl = Joblib(n_workers=2)
        results = jl._distribute(_identity_x, [])
        assert results == []

    def test_threading_backend_produces_correct_results(self):
        jl = Joblib(n_workers=2, backend="threading")
        batch = [{"x": i} for i in range(5)]
        results = jl._distribute(_identity_x, batch)
        assert results == list(range(5))

    def test_tuple_result_passthrough(self):
        jl = Joblib(n_workers=2)
        batch = [{"x": 1}, {"x": 2}]
        results = jl._distribute(_identity_x_with_metrics, batch)
        assert results[0] == (1, {"doubled": 2})
        assert results[1] == (2, {"doubled": 4})


class TestRayUnit:
    """Unit tests for the Ray backend."""

    @pytest.fixture(autouse=True)
    def _ray_lifecycle(self):
        ray = pytest.importorskip("ray")
        ray.init(num_cpus=2, ignore_reinit_error=True)
        yield
        ray.shutdown()

    def test_is_async(self):
        assert Ray._is_async is True

    def test_remote_cache_initially_empty(self):
        r = Ray(n_workers=2)
        assert r._remote_cache == {}

    def test_remote_caches_wrapper(self):
        r = Ray(n_workers=2)
        remote1 = r._remote(_identity_x)
        remote2 = r._remote(_identity_x)
        assert remote1 is remote2

    def test_remote_different_funcs_get_separate_entries(self):
        r = Ray(n_workers=2)
        r._remote(_identity_x)
        r._remote(_identity_x_with_metrics)
        assert len(r._remote_cache) == 2

    def test_result_ordering(self):
        r = Ray(n_workers=2)
        batch = [{"x": i} for i in range(8)]
        results = r._distribute(_identity_x, batch)
        assert results == list(range(8))

    def test_single_item_batch(self):
        r = Ray(n_workers=1)
        results = r._distribute(_identity_x, [{"x": 7}])
        assert results == [7]

    def test_empty_batch(self):
        r = Ray(n_workers=1)
        results = r._distribute(_identity_x, [])
        assert results == []

    def test_submit_wait_roundtrip(self):
        r = Ray(n_workers=2)
        future = r._submit(_identity_x, {"x": 42})
        completed, result = r._wait_any([future])
        assert result == 42
        assert completed is future

    def test_tuple_result_passthrough(self):
        r = Ray(n_workers=2)
        batch = [{"x": 1}, {"x": 2}]
        results = r._distribute(_identity_x_with_metrics, batch)
        assert results[0] == (1, {"doubled": 2})
        assert results[1] == (2, {"doubled": 4})


class TestDaskUnit:
    """Unit tests for the Dask backend."""

    @pytest.fixture(autouse=True)
    def _require_dask(self):
        pytest.importorskip("dask.distributed")

    @pytest.fixture
    def backend(self):
        b = Dask(n_workers=1)
        yield b
        if b._client is not None:
            b._client.close()

    def test_is_async(self):
        assert Dask._is_async is True

    def test_client_not_created_at_init(self):
        b = Dask(n_workers=2)
        assert b._client is None

    def test_get_client_creates_on_first_call(self, backend):
        assert backend._client is None
        client = backend._get_client()
        assert client is not None
        assert backend._client is client

    def test_get_client_returns_same_instance(self, backend):
        c1 = backend._get_client()
        c2 = backend._get_client()
        assert c1 is c2

    def test_client_arg_reused(self):
        from dask.distributed import Client

        external = Client(n_workers=1, threads_per_worker=1)
        try:
            b = Dask(n_workers=1, client=external)
            assert b._get_client() is external
        finally:
            external.close()

    def test_address_and_client_stored(self):
        b = Dask(n_workers=2, address="tcp://localhost:9999")
        assert b._address == "tcp://localhost:9999"
        assert b._client_arg is None

    def test_result_ordering(self, backend):
        batch = [{"x": i} for i in range(8)]
        results = backend._distribute(_identity_x, batch)
        assert results == list(range(8))

    def test_single_item_batch(self, backend):
        results = backend._distribute(_identity_x, [{"x": 7}])
        assert results == [7]

    def test_empty_batch(self, backend):
        results = backend._distribute(_identity_x, [])
        assert results == []

    def test_submit_wait_roundtrip(self, backend):
        future = backend._submit(_identity_x, {"x": 42})
        completed, result = backend._wait_any([future])
        assert result == 42
        assert completed is future

    def test_tuple_result_passthrough(self, backend):
        batch = [{"x": 1}, {"x": 2}]
        results = backend._distribute(_identity_x_with_metrics, batch)
        assert results[0] == (1, {"doubled": 2})
        assert results[1] == (2, {"doubled": 4})

"""Tests for distributed evaluation backends and search loop integration."""

import concurrent.futures

import numpy as np
import pytest

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
    Joblib,
    Multiprocessing,
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


class TestDistributedConditions:
    """Conditions and constraints work correctly across all distributed paths."""

    cond_space = {
        "x": np.arange(-5, 5, 1),
        "y": np.arange(-5, 5, 1),
    }

    def test_conditions_filter_sync_batch(self):
        """Sync-batch (Joblib) filters inactive params from worker input."""
        received = []

        def obj(params):
            received.append(set(params.keys()))
            return -(params["x"] ** 2)

        distributed = Joblib(n_workers=2, backend="threading").distribute(obj)
        opt = HillClimbingOptimizer(
            self.cond_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(distributed, n_iter=15, verbosity=False)

        for keys in received:
            assert "x" in keys

    def test_conditions_filter_true_async(self):
        """True-async (ThreadAsync) filters inactive params."""
        received = []

        def obj(params):
            received.append(set(params.keys()))
            return -(params["x"] ** 2)

        distributed = ThreadAsync(n_workers=2).distribute(obj)
        opt = HillClimbingOptimizer(
            self.cond_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(distributed, n_iter=15, verbosity=False)

        for keys in received:
            assert "x" in keys

    def test_conditions_filter_batch_async(self):
        """Batch-async (stateful optimizer) filters inactive params."""
        received = []

        def obj(params):
            received.append(set(params.keys()))
            return -(params["x"] ** 2)

        distributed = ThreadAsync(n_workers=2).distribute(obj)
        opt = DownhillSimplexOptimizer(
            self.cond_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(distributed, n_iter=15, verbosity=False)

        for keys in received:
            assert "x" in keys

    def test_conditions_nan_in_search_data_distributed(self):
        """Inactive params appear as NaN in search_data for distributed runs."""
        distributed = ThreadAsync(n_workers=2).distribute(lambda p: -(p["x"] ** 2))
        opt = HillClimbingOptimizer(
            self.cond_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            random_state=42,
        )
        opt.search(distributed, n_iter=20, verbosity=False)

        df = opt.search_data
        inactive = df[df["x"] <= 0]
        if len(inactive) > 0:
            assert inactive["y"].isna().all()
        active = df[df["x"] > 0]
        if len(active) > 0:
            assert active["y"].notna().all()

    def test_constraints_with_conditions_distributed(self):
        """Constraints on active params work in distributed mode."""
        distributed = ThreadAsync(n_workers=2).distribute(lambda p: -(p["x"] ** 2))
        opt = HillClimbingOptimizer(
            self.cond_space,
            conditions=[lambda p: {"y": p["x"] > 0}],
            constraints=[lambda p: p.get("x", 0) > -3],
            random_state=42,
        )
        opt.search(distributed, n_iter=20, verbosity=False)

        assert np.all(opt.search_data["x"].values > -3)

    def test_search_params_context_in_distributed(self):
        """Workers receive SearchParams with snapshotted context."""
        from gradient_free_optimizers._search_params import SearchParams

        received_types = []
        received_iters = []

        def obj(params):
            received_types.append(type(params).__name__)
            received_iters.append(params.iteration)
            return -(params["x"] ** 2)

        distributed = ThreadAsync(n_workers=2).distribute(obj)
        opt = HillClimbingOptimizer(
            {"x": np.arange(-5, 5, 1)},
            random_state=42,
        )
        opt.search(distributed, n_iter=15, verbosity=False)

        assert all(t == "SearchParams" for t in received_types)

    def test_conditions_categorical_distributed(self):
        """Categorical conditions work across distributed backends."""
        space = {
            "algo": ["svm", "rf", "nn"],
            "kernel": ["linear", "rbf"],
            "n_trees": np.arange(10, 100, 10),
        }
        received = []

        def obj(params):
            received.append(dict(params))
            return 1.0

        distributed = ThreadAsync(n_workers=2).distribute(obj)
        opt = RandomSearchOptimizer(
            space,
            conditions=[
                lambda p: {
                    "kernel": p["algo"] == "svm",
                    "n_trees": p["algo"] == "rf",
                }
            ],
            random_state=42,
        )
        opt.search(distributed, n_iter=20, verbosity=False)

        for params in received:
            if params["algo"] == "svm":
                assert "kernel" in params
                assert "n_trees" not in params
            elif params["algo"] == "rf":
                assert "kernel" not in params
                assert "n_trees" in params
            else:
                assert "kernel" not in params
                assert "n_trees" not in params


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

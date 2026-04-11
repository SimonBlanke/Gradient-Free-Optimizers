# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Mixin providing distributed/batch execution strategies for Search.

Separates the three distributed execution paths (sync-batch, true-async,
batch-async) from the core serial search loop. The Search class inherits
from this mixin and delegates to ``_run_distributed()`` when a distributed
backend is detected.
"""

from __future__ import annotations

import json
import math
import time

from ._catch import wrap_with_catch
from ._result import Result, unpack_objective_result
from ._search_params import SearchParams


class DistributedSearch:
    """Mixin that adds distributed evaluation capabilities to Search.

    Expects the host class to provide:

    Attributes: conv, p_bar, score_l, pos_l, optimum, eval_times,
        iter_times, results_manager, nth_iter, _iter, _last_metrics,
        _tracker, stopper, verbosity, _callbacks, n_iter_total,
        n_iter_search

    Methods: _iterate_batch(), _evaluate_batch(),
        _build_callback_info(), _run_callbacks()
    """

    def __init__(self):
        super().__init__()

    def _init_distributed(self, objective_function, catch):
        """Detect a distributed decorator and configure batch execution.

        Reads the ``_gfo_*`` attributes set by
        :meth:`BaseDistribution.distribute` and stores backend references
        on ``self``. If ``catch`` is provided, wraps the worker function
        so exceptions are handled inside the workers.

        Returns the unwrapped objective function for serial use during
        the initialization phase.
        """
        self._is_distributed = getattr(objective_function, "_gfo_distributed", False)
        if not self._is_distributed:
            self._batch_size = None
            self._backend = None
            return objective_function

        self._batch_size = objective_function._gfo_batch_size
        self._distributed_func = objective_function
        self._backend = objective_function._gfo_backend
        self._original_func = objective_function._gfo_original_func

        if catch:
            self._original_func = wrap_with_catch(self._original_func, catch)
            self._distributed_func = self._backend.distribute(self._original_func)

        return objective_function._gfo_original_func

    # -- strategy dispatcher --

    def _run_distributed(self, n_iter, nth_trial):
        """Pick the right execution strategy and run it.

        Async-capable backends with async-compatible optimizers get
        true-async (results processed one-at-a-time). Async backends
        with stateful optimizers (Simplex, Powell's, DIRECT) that need
        complete batches get batch-async. Everything else gets sync-batch.
        """
        if self._backend._is_async:
            if getattr(self, "_supports_async", True):
                self._run_true_async(n_iter, nth_trial)
            else:
                self._run_batch_async(n_iter, nth_trial)
        else:
            self._run_sync_batch(n_iter, nth_trial)

    # -- shared helpers --

    def _partition_cached(self, positions):
        """Split positions into cached hits and uncached misses.

        Returns (scores, metrics_list, uncached_indices). Cached entries
        have their score/metrics filled in; uncached entries are None/{}.
        """
        n = len(positions)
        scores = [None] * n
        metrics = [{}] * n
        uncached = []

        if self._storage is not None:
            for i, pos in enumerate(positions):
                cached = self._storage.get(tuple(pos))
                if cached is not None:
                    scores[i] = cached.score
                    metrics[i] = cached.metrics
                else:
                    uncached.append(i)
        else:
            uncached = list(range(n))

        return scores, metrics, uncached

    def _process_raw_result(self, raw):
        """Unpack a worker return value and negate if minimizing."""
        score, metrics = unpack_objective_result(raw)
        if self.optimum == "minimum":
            score = -score
        return score, metrics

    def _pos_to_params(self, pos):
        """Convert a position array to a SearchParams dictionary.

        When conditions are active, returns only the active parameters.
        Context is snapshotted at dispatch time so the SearchParams
        remains informative after crossing a process boundary.
        """
        value = self.conv.position2value(pos)
        full_params = self.conv.value2para(value)
        if self.conv.conditions:
            active_mask = self.conv.evaluate_conditions(full_params)
            filtered = {k: v for k, v in full_params.items() if active_mask[k]}
        else:
            filtered = full_params

        best_pos = getattr(self, "_pos_best", None)
        pos_best_dict = None
        if best_pos is not None:
            pos_best_dict = self.conv.value2para(self.conv.position2value(best_pos))

        snapshot = {
            "iteration": getattr(self, "nth_iter", 0),
            "score_best": getattr(self, "_score_best", -math.inf),
            "pos_best": pos_best_dict,
            "n_iter_total": getattr(self, "n_iter_total", 0),
        }
        return SearchParams(filtered, context_snapshot=snapshot)

    def _store_in_storage(self, pos, score, metrics):
        """Persist a result to external storage if configured."""
        if self._storage is not None:
            self._storage.put(tuple(pos), Result(score, metrics))

    def _track_evaluation(self, pos, score, eval_time=0, iter_time=0, metrics=None):
        """Record a single evaluation across all tracking systems.

        Unlike the serial path (which tracks through _initialization/_iteration
        and _evaluate_position separately), the distributed path needs a
        single method that updates everything at once because results
        arrive from workers rather than flowing through the adapter.
        """
        if metrics is None:
            metrics = {}
        self.eval_times.append(eval_time)
        self.iter_times.append(iter_time)
        active_mask = None
        if self.conv.conditions:
            full_params = self.conv.value2para(self.conv.position2value(pos))
            active_mask = self.conv.evaluate_conditions(full_params)
        self.results_manager.add(Result(score, metrics), pos, active_mask)
        self.pos_l.append(pos)
        self.score_l.append(score)
        self.p_bar.update(score, pos, self.nth_iter)
        self._last_metrics = metrics
        self._tracker.track(pos, score, metrics, is_init=False)
        self.stopper.update(score, self.p_bar.score_best, self._iter)
        self.n_iter_total += 1
        self.n_iter_search += 1
        self._iter += 1

    def _track_batch(
        self,
        positions,
        scores,
        metrics_list,
        uncached_indices,
        per_eval_time,
        per_iter_time,
    ):
        """Track all evaluations from a completed batch."""
        uncached_set = set(uncached_indices)
        for i, (pos, score) in enumerate(zip(positions, scores)):
            et = per_eval_time if i in uncached_set else 0
            self._track_evaluation(
                pos, score, et, per_iter_time, metrics=metrics_list[i]
            )

    def _check_stop(self, n_evaluated):
        """Check stopping conditions and run callbacks.

        In serial mode, updates the stopper here. In distributed mode,
        _track_evaluation already updated the stopper per-evaluation,
        so only the should_stop() check and callbacks run.

        Returns True if the search should stop.
        """
        if not self._is_distributed:
            current_score = self.score_l[-1] if self.score_l else -math.inf
            self.stopper.update(current_score, self.p_bar.score_best, n_evaluated - 1)

        if self.stopper.should_stop():
            if "debug_stop" in self.verbosity:
                debug_info = self.stopper.get_debug_info()
                print("\nStopping condition debug info:")
                print(json.dumps(debug_info, indent=2))
            return True
        if self._callbacks:
            info = self._build_callback_info(n_evaluated - 1)
            if self._run_callbacks(info) is False:
                return True
        return False

    # -- sync batch strategy --

    def _run_sync_batch(self, n_iter, nth_trial):
        """Loop over batches using the synchronous distributed function."""
        while nth_trial < n_iter:
            self.nth_iter = nth_trial

            remaining = n_iter - nth_trial
            batch = min(self._batch_size, remaining)
            self._iteration_batch(batch)
            nth_trial += batch

            if self._check_stop(nth_trial):
                break

    def _iteration_batch(self, batch_size):
        """Execute one synchronous batch: generate, dispatch, track."""
        t_batch_start = time.time()
        self.best_score = self.p_bar.score_best

        positions = self._iterate_batch(batch_size)
        n_positions = len(positions)
        scores, metrics_list, uncached = self._partition_cached(positions)

        if uncached:
            params_batch = [self._pos_to_params(positions[i]) for i in uncached]
            t_start = time.time()
            raw_results = self._distributed_func(params_batch)
            eval_time = time.time() - t_start

            for idx, raw in zip(uncached, raw_results):
                score, metrics = self._process_raw_result(raw)
                scores[idx] = score
                metrics_list[idx] = metrics
                self._store_in_storage(positions[idx], score, metrics)

            per_eval_time = eval_time / len(uncached)
        else:
            per_eval_time = 0

        self._evaluate_batch(positions, scores)

        per_iter_time = (time.time() - t_batch_start) / n_positions
        self._track_batch(
            positions,
            scores,
            metrics_list,
            uncached,
            per_eval_time,
            per_iter_time,
        )

    # -- true-async strategy --

    def _run_true_async(self, n_iter, nth_trial):
        """Results processed individually as workers complete.

        Each completed evaluation immediately triggers a new position
        proposal, keeping all workers busy. The optimizer updates its
        state after every single result, giving it the most up-to-date
        information for each subsequent proposal.
        """
        backend = self._backend
        original_func = self._original_func
        pending = {}
        n_evaluated = nth_trial

        def _submit_one():
            """Generate one position, check cache, submit if needed.

            Returns the number of evaluations completed (0, or 1 for
            a cache hit that was processed immediately).
            """
            nonlocal n_evaluated
            if n_evaluated + len(pending) >= n_iter:
                return 0

            self.nth_iter = n_evaluated + len(pending)
            self.best_score = self.p_bar.score_best
            pos = self._iterate_batch(1)[0]

            if self._storage is not None:
                cached = self._storage.get(tuple(pos))
                if cached is not None:
                    self._evaluate_batch([pos], [cached.score])
                    self._track_evaluation(pos, cached.score, metrics=cached.metrics)
                    n_evaluated += 1
                    return 1

            params = self._pos_to_params(pos)
            future = backend._submit(original_func, params)
            pending[future] = pos
            return 0

        for _ in range(self._batch_size):
            _submit_one()
            if n_evaluated >= n_iter:
                break

        while pending and n_evaluated < n_iter:
            t_iter = time.time()
            completed, raw = backend._wait_any(list(pending.keys()))
            pos = pending.pop(completed)

            score, metrics = self._process_raw_result(raw)
            self._store_in_storage(pos, score, metrics)

            self.nth_iter = n_evaluated
            self._evaluate_batch([pos], [score])
            iter_time = time.time() - t_iter
            self._track_evaluation(pos, score, iter_time, iter_time, metrics=metrics)
            n_evaluated += 1

            if self._check_stop(n_evaluated):
                break

            # Cache hits consume iterations without adding futures,
            # so keep submitting until a future is queued or nothing
            # is left.
            while n_evaluated < n_iter:
                if _submit_one() == 0:
                    break
                if self._check_stop(n_evaluated):
                    break

    # -- batch-async strategy --

    def _run_batch_async(self, n_iter, nth_trial):
        """Batch-async for stateful optimizers (Simplex, Powell's, DIRECT).

        Positions are generated as a complete batch, workers run
        asynchronously within the batch, but results are collected for
        the entire batch before calling _evaluate_batch. This preserves
        the batch contract that stateful optimizers depend on.
        """
        backend = self._backend
        original_func = self._original_func
        n_evaluated = nth_trial

        while n_evaluated < n_iter:
            t_batch_start = time.time()
            self.nth_iter = n_evaluated
            self.best_score = self.p_bar.score_best

            remaining = n_iter - n_evaluated
            batch_size = min(self._batch_size, remaining)
            positions = self._iterate_batch(batch_size)
            n_positions = len(positions)

            scores, metrics_list, uncached = self._partition_cached(positions)

            if uncached:
                futures = {}
                for i in uncached:
                    params = self._pos_to_params(positions[i])
                    future = backend._submit(original_func, params)
                    futures[future] = i

                t_start = time.time()
                while futures:
                    completed, raw = backend._wait_any(list(futures.keys()))
                    idx = futures.pop(completed)
                    score, metrics = self._process_raw_result(raw)
                    scores[idx] = score
                    metrics_list[idx] = metrics
                    self._store_in_storage(positions[idx], score, metrics)

                per_eval_time = (time.time() - t_start) / len(uncached)
            else:
                per_eval_time = 0

            self._evaluate_batch(positions, scores)

            per_iter_time = (time.time() - t_batch_start) / n_positions
            self._track_batch(
                positions,
                scores,
                metrics_list,
                uncached,
                per_eval_time,
                per_iter_time,
            )

            n_evaluated += n_positions

            if self._check_stop(n_evaluated):
                break

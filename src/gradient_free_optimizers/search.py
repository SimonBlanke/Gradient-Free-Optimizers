# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from ._callback import CallbackInfo
from ._catch import wrap_with_catch
from ._data import DataAccessor, SearchTracker
from ._fitness_mapper import FitnessMapper, ScalarIdentity, WeightedSum
from ._memory import CachedObjectiveAdapter
from ._objective_adapter import ObjectiveAdapter
from ._print_info import print_summary
from ._progress_bar import ProgressBarLVL0, ProgressBarLVL1
from ._result import (
    Result,
    negate_objectives,
    objectives_as_list,
    unpack_objective_result,
)
from ._results_manager import ResultsManager
from ._search_statistics import SearchStatistics
from ._stopping_conditions import OptimizationStopper
from ._times_tracker import TimesTracker
from .storage import BaseStorage

if TYPE_CHECKING:
    import pandas as pd


class Search(TimesTracker, SearchStatistics):
    """
    High-level interface for running optimization searches.

    Search provides the main entry point for optimization, handling the
    orchestration of initialization, iteration, progress tracking, memory
    caching, and result collection. It bridges the gap between user-facing
    API methods (like ``search()``) and the underlying optimizer mechanics.

    This class is designed as a mixin that combines with optimizer classes
    through multiple inheritance. It requires the optimizer to provide:

    - ``_init_pos()``: Generate initial positions
    - ``_iterate()``: Generate the next position to evaluate
    - ``_evaluate_init(score)``: Process initialization scores
    - ``_evaluate(score)``: Process iteration scores
    - ``_finish_initialization()``: Transition from init to iteration phase
    - ``conv``: Converter object for position/value/parameter transformations

    The search process follows this flow:

    1. **Initialization Phase**: Evaluate predefined initial positions
       (grid, random, vertices, warm start) to seed the optimizer
    2. **Iteration Phase**: Repeatedly generate and evaluate new positions
       using the optimizer's strategy
    3. **Termination**: Stop when n_iter reached or stopping condition met

    Attributes
    ----------
    optimizers : list
        List of optimizer instances (for ensemble/multi-optimizer support).
    search_data : pd.DataFrame
        DataFrame containing all evaluated positions and scores after search.
    best_score : float
        Best score found during the search.
    best_para : dict
        Parameters corresponding to the best score.
    best_value : array
        Raw parameter values corresponding to the best score.

    See Also
    --------
    TimesTracker : Tracks evaluation and iteration timing.
    SearchStatistics : Tracks search progress statistics.
    """

    def __init__(self):
        super().__init__()

        self.optimizers = []
        self.new_results_list = []
        self.all_results_list = []

        self.score_l = []
        self.pos_l = []
        self.random_seed = None

        self.results_manager = None  # Initialized in _init_search with converter
        self._search_data_cache = None  # Lazy DataFrame cache

        self._tracker: SearchTracker | None = None
        self.__data: DataAccessor | None = None
        self._n_objectives: int = 1
        self._fitness_mapper: FitnessMapper = ScalarIdentity()
        self._last_objectives: list[float] | None = None

    @TimesTracker.iter_time
    def _initialization(self):
        self.best_score = self.p_bar.score_best

        init_pos = self._init_pos()

        score_new = self._evaluate_position(init_pos)
        self._evaluate_init(score_new)

        self.pos_l.append(init_pos)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, init_pos, self.nth_iter)
        self._tracker.track(
            init_pos,
            score_new,
            self._last_metrics,
            is_init=True,
            objectives=self._last_objectives,
        )

        self.n_init_total += 1
        self.n_init_search += 1

    @TimesTracker.iter_time
    def _iteration(self):
        self.best_score = self.p_bar.score_best

        pos_new = self._iterate()

        score_new = self._evaluate_position(pos_new)
        self._evaluate(score_new)

        self.pos_l.append(pos_new)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, pos_new, self.nth_iter)
        self._tracker.track(
            pos_new,
            score_new,
            self._last_metrics,
            is_init=False,
            objectives=self._last_objectives,
        )

        self.n_iter_total += 1
        self.n_iter_search += 1

    def _iteration_batch(self, batch_size):
        t_batch_start = time.time()
        self.best_score = self.p_bar.score_best

        positions = self._iterate_batch(batch_size)
        n_positions = len(positions)

        scores = [None] * n_positions
        metrics_list = [{}] * n_positions
        objectives_list = [None] * n_positions
        uncached_indices = []

        if self._storage is not None:
            for i, pos in enumerate(positions):
                cached = self._storage.get(tuple(pos))
                if cached is not None:
                    scores[i] = cached.score
                    metrics_list[i] = cached.metrics
                    objectives_list[i] = cached.objectives
                else:
                    uncached_indices.append(i)
        else:
            uncached_indices = list(range(n_positions))

        if uncached_indices:
            params_batch = []
            for i in uncached_indices:
                value = self.conv.position2value(positions[i])
                params_batch.append(self.conv.value2para(value))

            t_start = time.time()
            raw_results = self._distributed_func(params_batch)
            eval_time = time.time() - t_start

            for idx, raw in zip(uncached_indices, raw_results):
                fitness, obj_list, metrics = self._process_raw_result(raw)
                scores[idx] = fitness
                metrics_list[idx] = metrics
                objectives_list[idx] = obj_list
                if self._storage is not None:
                    self._storage.put(
                        tuple(positions[idx]),
                        Result(fitness, metrics, obj_list),
                    )

            per_eval_time = eval_time / len(uncached_indices)
        else:
            per_eval_time = 0

        self._evaluate_batch(positions, scores)

        per_iter_time = (time.time() - t_batch_start) / n_positions

        uncached_set = set(uncached_indices)
        for i, (pos, score) in enumerate(zip(positions, scores)):
            et = per_eval_time if i in uncached_set else 0
            self._track_evaluation(
                pos,
                score,
                et,
                per_iter_time,
                metrics=metrics_list[i],
                objectives=objectives_list[i],
            )

    def _process_raw_result(self, raw):
        """Unpack a worker result, negate if minimizing, compute fitness.

        Consolidates the unpack / negate / fitness-map steps that
        previously were scattered across _iteration_batch, _run_true_async,
        and _run_batch_async.

        Returns
        -------
        tuple[float, list[float] | None, dict]
            (fitness, objectives_list, metrics)
        """
        objectives, metrics = unpack_objective_result(raw)
        if self.optimum == "minimum":
            objectives = negate_objectives(objectives)
        fitness = self._fitness_mapper(objectives)
        obj_list = objectives_as_list(objectives, self._n_objectives)
        return fitness, obj_list, metrics

    def _track_evaluation(
        self,
        pos,
        score,
        eval_time=0,
        iter_time=0,
        metrics=None,
        objectives=None,
    ):
        """Record a single evaluation result across all tracking systems."""
        if metrics is None:
            metrics = {}
        self.eval_times.append(eval_time)
        self.iter_times.append(iter_time)
        self.results_manager.add(Result(score, metrics, objectives), pos)
        self.pos_l.append(pos)
        self.score_l.append(score)
        self.p_bar.update(score, pos, self.nth_iter)
        self._last_metrics = metrics
        self._last_objectives = objectives
        self._tracker.track(
            pos,
            score,
            metrics,
            is_init=False,
            objectives=objectives,
        )
        self.stopper.update(score, self.p_bar.score_best, self._iter)
        self.n_iter_total += 1
        self.n_iter_search += 1
        self._iter += 1

    def _check_stop(self, n_evaluated):
        """Check stopping conditions and run callbacks. Returns True if should stop."""
        # In serial mode, the stopper needs per-evaluation updates here.
        # In distributed mode, _track_evaluation already updated the stopper
        # for each evaluation in the batch, giving correct granularity for
        # early_stopping counters.
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

    def _run_async_loop(self, n_iter, nth_trial):
        """Dispatch to true-async or batch-async based on optimizer capability."""
        if getattr(self, "_supports_async", True):
            self._run_true_async(n_iter, nth_trial)
        else:
            self._run_batch_async(n_iter, nth_trial)

    def _run_true_async(self, n_iter, nth_trial):
        """Async iteration: results processed individually as workers complete.

        Each completed evaluation immediately triggers a new position proposal,
        keeping all workers busy. The optimizer updates its state after every
        single result, giving it the most up-to-date information for each
        subsequent proposal.
        """
        backend = self._backend
        original_func = self._original_func
        pending = {}
        n_evaluated = nth_trial

        def _submit_one():
            """Generate one position, check cache, submit if needed.

            Returns the number of evaluations completed (0 or 1 for cache hit).
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
                    self._track_evaluation(
                        pos,
                        cached.score,
                        metrics=cached.metrics,
                        objectives=cached.objectives,
                    )
                    n_evaluated += 1
                    return 1

            value = self.conv.position2value(pos)
            params = self.conv.value2para(value)
            future = backend._submit(original_func, params)
            pending[future] = pos
            return 0

        # Fill initial worker slots
        for _ in range(self._batch_size):
            _submit_one()
            if n_evaluated >= n_iter:
                break

        # Process results as they arrive
        while pending and n_evaluated < n_iter:
            t_iter = time.time()
            completed, raw = backend._wait_any(list(pending.keys()))
            pos = pending.pop(completed)

            fitness, obj_list, metrics = self._process_raw_result(raw)

            if self._storage is not None:
                self._storage.put(
                    tuple(pos),
                    Result(fitness, metrics, obj_list),
                )

            self.nth_iter = n_evaluated
            self._evaluate_batch([pos], [fitness])
            iter_time = time.time() - t_iter
            self._track_evaluation(
                pos,
                fitness,
                iter_time,
                iter_time,
                metrics=metrics,
                objectives=obj_list,
            )
            n_evaluated += 1

            if self._check_stop(n_evaluated):
                break

            # Refill the freed worker slot. Cache hits consume iterations
            # without adding futures, so keep trying until a future is
            # queued or there's nothing left to submit.
            while n_evaluated < n_iter:
                if _submit_one() == 0:
                    break
                if self._check_stop(n_evaluated):
                    break

    def _run_batch_async(self, n_iter, nth_trial):
        """Batch-async iteration for stateful optimizers.

        Positions are generated as a complete batch via _iterate_batch(n),
        submitted to workers asynchronously, but results are collected for
        the entire batch before calling _evaluate_batch. This preserves
        the batch contract that stateful optimizers (Simplex, Powell's,
        DIRECT) depend on, while still benefiting from async worker
        scheduling within each batch.
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

            scores = [None] * n_positions
            metrics_list = [{}] * n_positions
            objectives_list = [None] * n_positions
            uncached_indices = []

            if self._storage is not None:
                for i, pos in enumerate(positions):
                    cached = self._storage.get(tuple(pos))
                    if cached is not None:
                        scores[i] = cached.score
                        metrics_list[i] = cached.metrics
                        objectives_list[i] = cached.objectives
                    else:
                        uncached_indices.append(i)
            else:
                uncached_indices = list(range(n_positions))

            if uncached_indices:
                futures = {}
                for i in uncached_indices:
                    value = self.conv.position2value(positions[i])
                    params = self.conv.value2para(value)
                    future = backend._submit(original_func, params)
                    futures[future] = i

                t_start = time.time()
                while futures:
                    completed, raw = backend._wait_any(list(futures.keys()))
                    idx = futures.pop(completed)

                    fitness, obj_list, metrics = self._process_raw_result(raw)

                    scores[idx] = fitness
                    metrics_list[idx] = metrics
                    objectives_list[idx] = obj_list
                    if self._storage is not None:
                        self._storage.put(
                            tuple(positions[idx]),
                            Result(fitness, metrics, obj_list),
                        )

                per_eval_time = (time.time() - t_start) / len(uncached_indices)
            else:
                per_eval_time = 0

            self._evaluate_batch(positions, scores)

            per_iter_time = (time.time() - t_batch_start) / n_positions

            uncached_set = set(uncached_indices)
            for i, (pos, score) in enumerate(zip(positions, scores)):
                et = per_eval_time if i in uncached_set else 0
                self._track_evaluation(
                    pos,
                    score,
                    et,
                    per_iter_time,
                    metrics=metrics_list[i],
                    objectives=objectives_list[i],
                )

            n_evaluated += n_positions

            if self._check_stop(n_evaluated):
                break

    def search(
        self,
        objective_function: Callable[[dict[str, Any]], float],
        n_iter: int,
        max_time: float | None = None,
        max_score: float | None = None,
        early_stopping: dict[str, Any] | None = None,
        memory: bool | BaseStorage = True,
        memory_warm_start: pd.DataFrame | None = None,
        verbosity: list[str] | Literal[False] = [
            "progress_bar",
            "print_results",
            "print_times",
        ],
        optimum: Literal["maximum", "minimum"] = "maximum",
        callbacks: list[Callable[[CallbackInfo], bool | None]] | None = None,
        catch: dict[type[Exception], int | float] | None = None,
        n_objectives: int = 1,
        fitness_mapper: FitnessMapper | None = None,
    ) -> None:
        """Run the optimization loop.

        Evaluates ``objective_function`` up to ``n_iter`` times, searching
        for the parameters that maximize (or minimize) the returned score.
        The search proceeds in two phases: an **initialization** phase that
        evaluates starting positions (controlled by the ``initialize``
        constructor parameter), followed by an **iteration** phase where the
        optimizer's strategy generates new candidate positions.

        After the search finishes, results are available via
        :attr:`best_para`, :attr:`best_score`, and :attr:`search_data`.

        Parameters
        ----------
        objective_function : callable
            The function to optimize. Must accept a single dictionary
            mapping parameter names to values and return either:

            - A ``float`` score, or
            - A tuple ``(float, dict)`` where the second element contains
              custom metrics (accessible via callbacks and ``search_data``).

            Example::

                def objective(params):
                    return -(params["x"] ** 2 + params["y"] ** 2)

                def objective_with_metrics(params):
                    loss = params["x"] ** 2
                    return -loss, {"loss": loss}

        n_iter : int
            Total number of iterations (including initialization).
            Each iteration evaluates the objective function once (unless
            a cached result is found when ``memory=True``).
        max_time : float or None, default=None
            Maximum wall-clock time in seconds. The search stops after
            the current iteration if the elapsed time exceeds this limit.
            ``None`` means no time limit.
        max_score : float or None, default=None
            Target score threshold. The search stops when the best score
            found so far reaches or exceeds this value. When
            ``optimum="minimum"``, this refers to the original (non-negated)
            score.
            ``None`` means no score limit.
        early_stopping : dict or None, default=None
            Configuration for stopping the search when progress stalls.
            ``None`` disables early stopping. When provided, the dictionary
            supports the following keys:

            - ``"n_iter_no_change"`` (int, required): Stop if no improvement
              is observed for this many consecutive iterations.
            - ``"tol_abs"`` (float, optional): Minimum absolute improvement
              required over the window to count as progress.
            - ``"tol_rel"`` (float, optional): Minimum relative improvement
              (in percent) required over the window to count as progress.

            Example::

                early_stopping = {"n_iter_no_change": 50}
                early_stopping = {"n_iter_no_change": 30, "tol_abs": 0.001}

        memory : bool or BaseStorage, default=True
            Controls evaluation caching. When ``True``, uses an in-memory
            dictionary (equivalent to ``MemoryStorage()``). When ``False``,
            disables caching entirely. A
            :class:`~gradient_free_optimizers.storage.BaseStorage` instance
            enables custom storage backends::

                from gradient_free_optimizers.storage import SQLiteStorage
                opt.search(objective, memory=SQLiteStorage("results.db"))

            ``SQLiteStorage`` persists results to disk, enabling crash
            recovery and cache reuse across runs. Works with distributed
            evaluation (positions are checked against the cache before
            being dispatched to workers).

            In-memory caching is especially useful for discrete search
            spaces where revisits are common.
        memory_warm_start : pd.DataFrame or None, default=None
            A DataFrame from a previous search (typically obtained via
            :attr:`search_data`) to pre-populate the evaluation cache.
            The DataFrame must contain columns matching the search space
            parameter names plus a ``"score"`` column. Requires
            ``memory=True``.

            Example::

                opt1 = HillClimbingOptimizer(search_space)
                opt1.search(objective, n_iter=50)

                opt2 = HillClimbingOptimizer(search_space)
                opt2.search(objective, n_iter=50,
                            memory_warm_start=opt1.search_data)

        verbosity : list[str] or False, default=\
["progress_bar", "print_results", "print_times"]
            Controls console output during and after the search.
            Pass ``False`` or an empty list for silent operation.

            Supported values:

            - ``"progress_bar"``: Show a live ``tqdm`` progress bar during
              the search.
            - ``"print_results"``: Print best score and best parameters
              after the search completes.
            - ``"print_times"``: Print timing breakdown (evaluation time,
              optimization overhead, throughput) after the search completes.
            - ``"print_search_stats"``: Print search statistics including
              iteration counts, acceptance rate, number of improvements,
              and longest plateau.
            - ``"print_statistics"``: Print score statistics (min, max,
              mean, std) after the search completes.
            - ``"debug_stop"``: Print detailed stopping condition debug
              info when the search terminates early.

        optimum : {"maximum", "minimum"}, default="maximum"
            Whether to maximize or minimize the objective function.
            When set to ``"minimum"``, the objective function's return
            value is negated internally so that the optimizer always
            maximizes. The reported ``best_score`` is in original
            (non-negated) units.
        callbacks : list[callable] or None, default=None
            A list of callback functions invoked after each iteration.
            Each callback receives a single argument ``info`` with the
            following attributes:

            - ``info.iteration`` (int): Current iteration index (0-based).
            - ``info.score`` (float): Score from the current evaluation.
            - ``info.params`` (dict): Parameters evaluated in this iteration.
            - ``info.best_score`` (float): Best score found so far.
            - ``info.best_para`` (dict): Parameters of the best score.
            - ``info.n_iter`` (int): Total iterations planned.
            - ``info.phase`` (str): ``"init"`` or ``"iter"``.
            - ``info.elapsed_time`` (float): Seconds since search started.
            - ``info.metrics`` (dict): Custom metrics from the objective
              function (empty if the objective returns only a score).
            - ``info.convergence`` (list[float]): Best score at each
              iteration so far.

            If any callback returns ``False``, the search stops
            immediately. Any other return value (including ``None``) is
            ignored and the search continues.

            Example::

                def log_progress(info):
                    if info.iteration % 10 == 0:
                        print(f"Iter {info.iteration}: best={info.best_score:.4f}")

                def stop_early(info):
                    if info.best_score > 0.99:
                        return False  # stops the search

                opt.search(objective, n_iter=100,
                           callbacks=[log_progress, stop_early])

        catch : dict[type, float] or None, default=None
            Error handling for the objective function. Maps exception
            types to fallback scores. When the objective function raises
            a caught exception, the optimizer records the fallback score
            instead of crashing. Exception subclasses are matched via
            ``isinstance``, so ``{Exception: ...}`` catches all.

            The fallback score is in the user's original units (before
            any negation from ``optimum="minimum"``). Use
            ``float('nan')`` or ``float('inf')`` to mark positions as
            invalid without inventing an artificial score.

            Example::

                catch = {ValueError: -1000, RuntimeError: float('nan')}

                opt.search(objective, n_iter=100, catch=catch)

        Examples
        --------
        Basic usage with default settings:

        >>> import numpy as np
        >>> from gradient_free_optimizers import HillClimbingOptimizer
        >>> def objective(para):
        ...     return -(para["x"] ** 2)
        >>> search_space = {"x": np.linspace(-10, 10, 100)}
        >>> opt = HillClimbingOptimizer(search_space)
        >>> opt.search(objective, n_iter=30)

        Using multiple stopping conditions:

        >>> opt.search(
        ...     objective,
        ...     n_iter=1000,
        ...     max_time=60,
        ...     max_score=-0.01,
        ...     early_stopping={"n_iter_no_change": 50},
        ... )
        """
        self.optimum = optimum
        self._callbacks = callbacks or []
        self._n_objectives = n_objectives
        if fitness_mapper is not None:
            self._fitness_mapper = fitness_mapper
        elif n_objectives > 1:
            self._fitness_mapper = WeightedSum(n_objectives=n_objectives)
        else:
            self._fitness_mapper = ScalarIdentity()

        self._init_search(
            objective_function,
            n_iter,
            max_time,
            max_score,
            early_stopping,
            memory,
            memory_warm_start,
            verbosity,
            catch,
        )

        nth_trial = 0

        # Initialization phase (always serial, even in distributed mode)
        while nth_trial < self.n_inits_norm and nth_trial < n_iter:
            self._search_step(nth_trial)

            current_score = self.score_l[-1] if self.score_l else -math.inf
            self.stopper.update(current_score, self.p_bar.score_best, nth_trial)

            if self._callbacks:
                info = self._build_callback_info(nth_trial)
                if self._run_callbacks(info) is False:
                    nth_trial = n_iter
                    break
            if self.stopper.should_stop():
                if "debug_stop" in self.verbosity:
                    debug_info = self.stopper.get_debug_info()
                    print("\nStopping condition debug info:")
                    print(json.dumps(debug_info, indent=2))
                nth_trial = n_iter
                break

            nth_trial += 1

        # Transition from init to iteration phase
        if nth_trial < n_iter and nth_trial == self.n_init_search:
            self._finish_initialization()

        # Iteration phase: async backends get their own loop,
        # sync backends use the original batch/serial loop
        if self._is_distributed and self._backend._is_async and nth_trial < n_iter:
            self._run_async_loop(n_iter, nth_trial)
        else:
            while nth_trial < n_iter:
                self.nth_iter = nth_trial

                if self._is_distributed:
                    remaining = n_iter - nth_trial
                    batch = min(self._batch_size, remaining)
                    self._iteration_batch(batch)
                    nth_trial += batch
                else:
                    self._iteration()
                    nth_trial += 1

                if self._check_stop(nth_trial):
                    break

        self._finish_search()

    def _evaluate_position(self, pos: list[int]) -> float:
        t = time.time()
        result, params = self.adapter(pos)
        self.eval_times.append(time.time() - t)
        self.results_manager.add(result, pos)
        self._last_metrics = result.metrics if result.metrics else {}
        self._last_objectives = result.objectives
        self._iter += 1
        return result.score

    @SearchStatistics.init_stats
    def _init_search(
        self,
        objective_function: Callable[[dict[str, Any]], float],
        n_iter: int,
        max_time: float | None,
        max_score: float | None,
        early_stopping: dict[str, Any] | None,
        memory: bool | BaseStorage,
        memory_warm_start: pd.DataFrame | None,
        verbosity: list[str] | Literal[False],
        catch: dict[type[Exception], int | float] | None = None,
    ) -> None:
        # Detect distributed decorator before any wrapping
        self._is_distributed = getattr(objective_function, "_gfo_distributed", False)
        if self._is_distributed:
            self._batch_size = objective_function._gfo_batch_size
            self._distributed_func = objective_function
            self._backend = objective_function._gfo_backend
            self._original_func = objective_function._gfo_original_func
            # Extract original function for single-point use during init
            objective_function = objective_function._gfo_original_func

            if catch:
                self._original_func = wrap_with_catch(self._original_func, catch)
                self._distributed_func = self._backend.distribute(self._original_func)
        else:
            self._batch_size = None
            self._backend = None

        if catch:
            objective_function = wrap_with_catch(objective_function, catch)

        if getattr(self, "optimum", "maximum") == "minimum":
            _obj = objective_function

            def _negate(params):
                out = _obj(params)
                objectives, metrics = unpack_objective_result(out)
                return (negate_objectives(objectives), metrics)

            self.objective_function = _negate
        else:
            self.objective_function = objective_function
        self.n_iter = n_iter
        self.max_time = max_time
        self.max_score = max_score
        self.early_stopping = early_stopping
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.verbosity = verbosity

        self._iter = 0
        self._last_metrics: dict = {}

        self._tracker = SearchTracker(self.conv)
        self._tracker.optimizer_name = self.__class__.__name__
        self._tracker.objective_name = getattr(
            objective_function, "__name__", str(objective_function)
        )
        self._tracker.random_seed = self.random_seed
        self.__data = None

        # Initialize ResultsManager only if not already created
        # (preserves results across searches)
        # Using lazy DataFrame construction to reduce memory for high-dimensional spaces
        if self.results_manager is None:
            self.results_manager = ResultsManager(self.conv)

        # Invalidate cached DataFrame since new results will be added
        self._search_data_cache = None

        if self.verbosity is False:
            self.verbosity = []

        start_time = time.time()
        self.stopper = OptimizationStopper(
            start_time=start_time,
            max_time=max_time,
            max_score=max_score,
            early_stopping=early_stopping,
        )

        if "progress_bar" in self.verbosity:
            self.p_bar = ProgressBarLVL1(
                self.nth_process, self.n_iter, self.objective_function
            )
        else:
            self.p_bar = ProgressBarLVL0(
                self.nth_process, self.n_iter, self.objective_function
            )

        _fm = self._fitness_mapper
        _no = self._n_objectives
        if isinstance(memory, BaseStorage):
            self.adapter = CachedObjectiveAdapter(
                self.conv,
                self.objective_function,
                storage=memory,
                fitness_mapper=_fm,
                n_objectives=_no,
            )
            self.adapter.memory(memory_warm_start)
            self._storage = self.adapter._storage
        elif memory not in [False, None]:
            self.adapter = CachedObjectiveAdapter(
                self.conv,
                self.objective_function,
                fitness_mapper=_fm,
                n_objectives=_no,
            )
            self.adapter.memory(memory_warm_start, memory)
            self._storage = self.adapter._storage
        else:
            self._storage = None
            self.adapter = ObjectiveAdapter(
                self.conv,
                self.objective_function,
                fitness_mapper=_fm,
                n_objectives=_no,
            )

        self.n_inits_norm = min((self.init.n_inits - self.n_init_total), self.n_iter)

    def _finish_search(self) -> None:
        # Don't construct DataFrame here - it's built lazily via search_data property
        # This avoids memory spike for high-dimensional search spaces
        self._search_data_cache = None

        self.best_score = self.p_bar.score_best
        self.best_value = self.conv.position2value(self.p_bar.pos_best)
        self.best_para = self.conv.value2para(self.best_value)

        self.p_bar.close()

        print_sections = {v for v in self.verbosity if v.startswith("print_")}
        if print_sections:
            print_summary(self._data, print_sections)

    @property
    def _data(self) -> DataAccessor:
        """Access search data and computed metrics (internal, may change).

        Available after calling ``search()``. Returns a
        :class:`~gradient_free_optimizers._data.data_accessor.DataAccessor`
        object with properties like ``best_score``, ``convergence_data``,
        ``overhead_pct``, and a
        :class:`~gradient_free_optimizers._data.raw_data.RawData`
        sub-accessor for internal tracking lists.
        """
        if self._tracker is None:
            raise AttributeError("Search data not available. Call search() first.")
        if self.__data is None:
            self.__data = DataAccessor(self._tracker, self)
        return self.__data

    @property
    def search_data(self) -> pd.DataFrame:
        """Lazily construct and return the search results DataFrame.

        The DataFrame is only built when this property is accessed, avoiding
        a large memory spike at the end of high-dimensional optimizations.
        The result is cached so subsequent accesses don't rebuild it.
        """
        if self._search_data_cache is None:
            self._search_data_cache = self.results_manager.dataframe
        return self._search_data_cache

    @search_data.setter
    def search_data(self, value: pd.DataFrame) -> None:
        """Allow direct assignment for backward compatibility."""
        self._search_data_cache = value

    @property
    def pareto_front(self) -> pd.DataFrame:
        """Non-dominated solutions from a multi-objective search.

        Computes the Pareto front from stored objective vectors. Only
        meaningful when ``n_objectives > 1`` was passed to ``search()``.
        Returns a subset of ``search_data`` containing only the rows
        that are not dominated by any other evaluated solution.

        Raises ``ValueError`` if no objectives were tracked (single-
        objective search or search not yet run).
        """
        data = self.search_data
        obj_cols = [c for c in data.columns if c.startswith("objective_")]
        if not obj_cols:
            raise ValueError(
                "No objective columns found. Use n_objectives > 1 in search()."
            )

        obj_values = data[obj_cols].values
        n = len(obj_values)
        is_dominated = [False] * n

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # j dominates i if j is >= in all objectives and > in at least one
                if all(obj_values[j] >= obj_values[i]) and any(
                    obj_values[j] > obj_values[i]
                ):
                    is_dominated[i] = True
                    break

        mask = [not d for d in is_dominated]
        return data.loc[mask].reset_index(drop=True)

    def _build_callback_info(self, nth_iter: int) -> CallbackInfo:
        pos = self.pos_l[-1]
        value = self.conv.position2value(list(pos))
        params = self.conv.value2para(value)

        if self.p_bar.pos_best is not None:
            best_value = self.conv.position2value(self.p_bar.pos_best)
            best_para = self.conv.value2para(best_value)
        else:
            best_para = {}

        return CallbackInfo(
            iteration=nth_iter,
            score=self.score_l[-1],
            params=params,
            best_score=self.p_bar.score_best,
            best_para=best_para,
            n_iter=self.n_iter,
            phase="init" if nth_iter < self.n_inits_norm else "iter",
            elapsed_time=time.time() - self.stopper.start_time,
            metrics=self._last_metrics,
            convergence=list(self._tracker.convergence),
            objectives=self._last_objectives,
        )

    def _run_callbacks(self, info: CallbackInfo) -> bool | None:
        for callback in self._callbacks:
            result = callback(info)
            if result is False:
                return False
        return None

    def _search_step(self, nth_iter: int) -> None:
        self.nth_iter = nth_iter

        if self.nth_iter < self.n_inits_norm:
            self._initialization()

        if self.nth_iter == self.n_init_search:
            self._finish_initialization()

        if self.n_init_search <= self.nth_iter < self.n_iter:
            self._iteration()

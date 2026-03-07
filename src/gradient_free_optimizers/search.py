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
from ._data import DataAccessor, SearchTracker
from ._memory import CachedObjectiveAdapter
from ._objective_adapter import ObjectiveAdapter
from ._print_info import print_summary
from ._progress_bar import ProgressBarLVL0, ProgressBarLVL1
from ._results_manager import ResultsManager
from ._search_statistics import SearchStatistics
from ._stopping_conditions import OptimizationStopper
from ._times_tracker import TimesTracker

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

    @TimesTracker.iter_time
    def _initialization(self):
        self.best_score = self.p_bar.score_best

        init_pos = self._init_pos()

        score_new = self._evaluate_position(init_pos)
        self._evaluate_init(score_new)

        self.pos_l.append(init_pos)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, init_pos, self.nth_iter)
        self._tracker.track(init_pos, score_new, self._last_metrics, is_init=True)

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
        self._tracker.track(pos_new, score_new, self._last_metrics, is_init=False)

        self.n_iter_total += 1
        self.n_iter_search += 1

    def search(
        self,
        objective_function: Callable[[dict[str, Any]], float],
        n_iter: int,
        max_time: float | None = None,
        max_score: float | None = None,
        early_stopping: dict[str, Any] | None = None,
        memory: bool = True,
        memory_warm_start: pd.DataFrame | None = None,
        verbosity: list[str] | Literal[False] = [
            "progress_bar",
            "print_results",
            "print_times",
        ],
        optimum: Literal["maximum", "minimum"] = "maximum",
        callbacks: list[Callable[[CallbackInfo], bool | None]] | None = None,
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

        memory : bool, default=True
            If ``True``, cache objective function evaluations in an
            in-memory dictionary keyed by position. When the optimizer
            revisits a previously evaluated position, the cached score is
            returned without calling the objective function again. This
            is especially useful for discrete search spaces where
            revisits are common.
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
        self._init_search(
            objective_function,
            n_iter,
            max_time,
            max_score,
            early_stopping,
            memory,
            memory_warm_start,
            verbosity,
        )

        for nth_trial in range(n_iter):
            self._search_step(nth_trial)

            # Update stopper with current state
            current_score = self.score_l[-1] if self.score_l else -math.inf
            best_score = self.p_bar.score_best
            self.stopper.update(current_score, best_score, nth_trial)

            if self._callbacks:
                info = self._build_callback_info(nth_trial)
                if self._run_callbacks(info) is False:
                    break

            if self.stopper.should_stop():
                # Log debugging information when stopping
                if "debug_stop" in self.verbosity:
                    debug_info = self.stopper.get_debug_info()
                    print("\nStopping condition debug info:")
                    print(json.dumps(debug_info, indent=2))
                break

        self._finish_search()

    def _evaluate_position(self, pos: list[int]) -> float:
        t = time.time()
        result, params = self.adapter(pos)
        self.eval_times.append(time.time() - t)
        # Store position instead of params dict for memory efficiency
        # Params are reconstructed lazily when search_data DataFrame is accessed
        self.results_manager.add(result, pos)
        self._last_metrics = result.metrics if result.metrics else {}
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
        memory: bool,
        memory_warm_start: pd.DataFrame | None,
        verbosity: list[str] | Literal[False],
    ) -> None:
        if getattr(self, "optimum", "maximum") == "minimum":
            self.objective_function = lambda pos: -objective_function(pos)
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

        if self.memory not in [False, None]:
            self.adapter = CachedObjectiveAdapter(self.conv, self.objective_function)
            self.adapter.memory(memory_warm_start, memory)
        else:
            self.adapter = ObjectiveAdapter(self.conv, self.objective_function)

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

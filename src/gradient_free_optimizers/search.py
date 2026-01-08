# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import json
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

from ._progress_bar import ProgressBarLVL0, ProgressBarLVL1
from ._times_tracker import TimesTracker
from ._search_statistics import SearchStatistics
from ._print_info import print_info
from ._stop_run import StopRun
from ._results_manager import ResultsManager
from ._objective_adapter import ObjectiveAdapter
from ._memory import CachedObjectiveAdapter
from ._stopping_conditions import OptimizationStopper

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

    - ``init_pos()``: Generate initial positions
    - ``iterate()``: Generate the next position to evaluate
    - ``evaluate_init(score)``: Process initialization scores
    - ``evaluate(score)``: Process iteration scores
    - ``finish_initialization()``: Transition from init to iteration phase
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

        self.results_manager = ResultsManager()

    @TimesTracker.eval_time
    def _score(self, pos):
        return self.score(pos)

    @TimesTracker.iter_time
    def _initialization(self):
        self.best_score = self.p_bar.score_best

        init_pos = self.init_pos()

        score_new = self._evaluate_position(init_pos)
        self.evaluate_init(score_new)

        self.pos_l.append(init_pos)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, init_pos, self.nth_iter)

        self.n_init_total += 1
        self.n_init_search += 1

    @TimesTracker.iter_time
    def _iteration(self):
        self.best_score = self.p_bar.score_best

        pos_new = self.iterate()

        score_new = self._evaluate_position(pos_new)
        self.evaluate(score_new)

        self.pos_l.append(pos_new)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, pos_new, self.nth_iter)

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
        verbosity: list[str] | Literal[False] = ["progress_bar", "print_results", "print_times"],
        optimum: Literal["maximum", "minimum"] = "maximum",
    ) -> None:
        self.optimum = optimum
        self.init_search(
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
            self.search_step(nth_trial)

            # Update stopper with current state
            current_score = self.score_l[-1] if self.score_l else -math.inf
            best_score = self.p_bar.score_best
            self.stopper.update(current_score, best_score, nth_trial)

            if self.stopper.should_stop():
                # Log debugging information when stopping
                if "debug_stop" in self.verbosity:
                    debug_info = self.stopper.get_debug_info()
                    print("\nStopping condition debug info:")
                    print(json.dumps(debug_info, indent=2))
                break

        self.finish_search()

    def _evaluate_position(self, pos: list[int]) -> float:
        result, params = self.adapter(pos)
        self.results_manager.add(result, params)
        self._iter += 1
        return result.score

    @SearchStatistics.init_stats
    def init_search(
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
            self.adapter = CachedObjectiveAdapter(self.conv, objective_function)
            self.adapter.memory(memory_warm_start, memory)
        else:
            self.adapter = ObjectiveAdapter(self.conv, objective_function)

        self.n_inits_norm = min((self.init.n_inits - self.n_init_total), self.n_iter)

    def finish_search(self) -> None:
        self.search_data = self.results_manager.dataframe

        self.best_score = self.p_bar.score_best
        self.best_value = self.conv.position2value(self.p_bar.pos_best)
        self.best_para = self.conv.value2para(self.best_value)
        """
        if self.memory not in [False, None]:
            self.memory_dict = self.mem.memory_dict
        else:
            self.memory_dict = {}
        """
        self.p_bar.close()

        print_info(
            self.verbosity,
            self.objective_function,
            self.best_score,
            self.best_para,
            self.eval_times,
            self.iter_times,
            self.n_iter,
            self.random_seed,
        )

    def search_step(self, nth_iter: int) -> None:
        self.nth_iter = nth_iter

        if self.nth_iter < self.n_inits_norm:
            self._initialization()

        if self.nth_iter == self.n_init_search:
            self.finish_initialization()

        if self.n_init_search <= self.nth_iter < self.n_iter:
            self._iteration()

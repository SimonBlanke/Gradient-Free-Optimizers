# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import json
import time
import numpy as np

from ._progress_bar import ProgressBarLVL0, ProgressBarLVL1
from ._times_tracker import TimesTracker
from ._search_statistics import SearchStatistics
from ._print_info import print_info
from ._stop_run import StopRun
from ._results_manager import ResultsManager
from ._objective_adapter import ObjectiveAdapter
from ._memory import CachedObjectiveAdapter
from ._stopping_conditions import OptimizationStopper


class Search(TimesTracker, SearchStatistics):
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
        objective_function,
        n_iter,
        max_time=None,
        max_score=None,
        early_stopping=None,
        memory=True,
        memory_warm_start=None,
        verbosity=["progress_bar", "print_results", "print_times"],
        optimum="maximum",
    ):
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
            current_score = self.score_l[-1] if self.score_l else -np.inf
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
        objective_function,
        n_iter,
        max_time,
        max_score,
        early_stopping,
        memory,
        memory_warm_start,
        verbosity,
    ):
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

    def finish_search(self):
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

    def search_step(self, nth_iter):
        self.nth_iter = nth_iter

        if self.nth_iter < self.n_inits_norm:
            self._initialization()

        if self.nth_iter == self.n_init_search:
            self.finish_initialization()

        if self.n_init_search <= self.nth_iter < self.n_iter:
            self._iteration()

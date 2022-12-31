# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from multiprocessing.managers import DictProxy

from .progress_bar import ProgressBarLVL0, ProgressBarLVL1
from .times_tracker import TimesTracker
from .search_statistics import SearchStatistics
from .memory import Memory
from .print_info import print_info
from .stop_run import StopRun

from .results_manager import ResultsManager


class Search(TimesTracker, SearchStatistics):
    def __init__(self):
        super().__init__()

        self.optimizers = []
        self.new_results_list = []
        self.all_results_list = []

        self.score_l = []
        self.pos_l = []
        self.random_seed = None

        self.search_state = "init"

        self.results_mang = ResultsManager()

    @TimesTracker.eval_time
    def _score(self, pos):
        return self.score(pos)

    @TimesTracker.iter_time
    def _initialization(self):
        self.best_score = self.p_bar.score_best

        init_pos = self.init_pos()

        score_new = self._score(init_pos)
        self.evaluate_init(score_new)

        self.pos_l.append(init_pos)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, init_pos, self.nth_iter)

        self.n_init_total += 1
        self.n_init_search += 1

        self.stop.update(self.p_bar.score_best, self.score_l)

    @TimesTracker.iter_time
    def _iteration(self):
        self.best_score = self.p_bar.score_best

        pos_new = self.iterate()

        score_new = self._score(pos_new)
        self.evaluate(score_new)

        self.pos_l.append(pos_new)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, pos_new, self.nth_iter)

        self.n_iter_total += 1
        self.n_iter_search += 1

        self.stop.update(self.p_bar.score_best, self.score_l)

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
    ):
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

        for nth_iter in range(n_iter):
            self.search_step(nth_iter)
            if self.stop.check():
                break

        self.finish_search()

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
        self.objective_function = objective_function
        self.n_iter = n_iter
        self.max_time = max_time
        self.max_score = max_score
        self.early_stopping = early_stopping
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.verbosity = verbosity

        self.results_mang.conv = self.conv

        if self.verbosity is False:
            self.verbosity = []

        start_time = time.time()
        self.stop = StopRun(
            start_time, self.max_time, self.max_score, self.early_stopping
        )

        if "progress_bar" in self.verbosity:
            self.p_bar = ProgressBarLVL1(
                self.nth_process, self.n_iter, self.objective_function
            )
        else:
            self.p_bar = ProgressBarLVL0(
                self.nth_process, self.n_iter, self.objective_function
            )

        if isinstance(self.memory, DictProxy):
            self.mem = Memory(self.memory_warm_start, self.conv, dict_proxy=self.memory)
            self.score = self.results_mang.score(
                self.mem.memory(self.objective_function)
            )
        elif self.memory is True:
            self.mem = Memory(self.memory_warm_start, self.conv)
            self.score = self.results_mang.score(
                self.mem.memory(self.objective_function)
            )
        else:
            self.score = self.results_mang.score(self.objective_function)

        self.n_inits_norm = min((self.init.n_inits - self.n_init_total), self.n_iter)

    def finish_search(self):
        self.search_data = self.results_mang.search_data

        self.best_score = self.p_bar.score_best
        self.best_value = self.conv.position2value(self.p_bar.pos_best)
        self.best_para = self.conv.value2para(self.best_value)

        if self.memory not in [False, None]:
            self.memory_dict = self.mem.memory_dict
        else:
            self.memory_dict = {}

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

        """
        # loop to initialize N positions
        for nth_iter in range(self.n_inits_norm):
            if self.stop.check(self.start_time, self.p_bar.score_best, self.score_l):
                break
            self._initialization(nth_iter)

        self.finish_initialization()

        # loop to do the iterations
        for nth_iter in range(self.n_init_search, self.n_iter):
            if self.stop.check(self.start_time, self.p_bar.score_best, self.score_l):
                break
            self._iteration(nth_iter)
        """

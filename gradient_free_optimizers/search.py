# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random

import numpy as np
import pandas as pd

from multiprocessing.managers import DictProxy

from .progress_bar import ProgressBarLVL0, ProgressBarLVL1
from .times_tracker import TimesTracker
from .memory import Memory
from .print_info import print_info
from .init_positions import Initializer


def time_exceeded(start_time, max_time):
    run_time = time.time() - start_time
    return max_time and run_time > max_time


def score_exceeded(score_best, max_score):
    return max_score and score_best >= max_score


def no_change(score_new_list, early_stopping):
    if "n_iter_no_change" not in early_stopping:
        print(
            "Warning n_iter_no_change-parameter must be set in order for early stopping to work"
        )
        return False

    n_iter_no_change = early_stopping["n_iter_no_change"]
    if len(score_new_list) <= n_iter_no_change:
        return False

    scores_np = np.array(score_new_list)

    max_score = max(score_new_list)
    max_index = np.argmax(scores_np)
    length_pos = len(score_new_list)

    diff = length_pos - max_index

    if diff > n_iter_no_change:
        return True

    first_n = length_pos - n_iter_no_change
    scores_first_n = score_new_list[:first_n]

    max_first_n = max(scores_first_n)

    if "tol_abs" in early_stopping and early_stopping["tol_abs"] is not None:
        tol_abs = early_stopping["tol_abs"]

        if abs(max_first_n - max_score) < tol_abs:
            return True

    if "tol_rel" in early_stopping and early_stopping["tol_rel"] is not None:
        tol_rel = early_stopping["tol_rel"]

        percent_imp = ((max_score - max_first_n) / abs(max_first_n)) * 100
        if percent_imp < tol_rel:
            return True


def set_random_seed(nth_process, random_state):
    """
    Sets the random seed separately for each thread
    (to avoid getting the same results in each thread)
    """
    if nth_process is None:
        nth_process = 0

    if random_state is None:
        random_state = np.random.randint(0, high=2 ** 31 - 2, dtype=np.int64)

    random.seed(random_state + nth_process)
    np.random.seed(random_state + nth_process)


class Search(TimesTracker):
    def __init__(self):
        super().__init__()

        self.optimizers = []
        self.new_results_list = []
        self.all_results_list = []

        self.score_l = []
        self.pos_l = []

    @TimesTracker.eval_time
    def _score(self, pos):
        return self.score(pos)

    @TimesTracker.iter_time
    def _initialization(self, init_pos, nth_iter):
        self.nth_iter = nth_iter
        self.best_score = self.p_bar.score_best

        self.init_pos(init_pos)

        score_new = self._score(init_pos)
        self.evaluate(score_new)

        self.pos_l.append(init_pos)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, init_pos, nth_iter)

    @TimesTracker.iter_time
    def _iteration(self, nth_iter):
        self.nth_iter = nth_iter
        self.best_score = self.p_bar.score_best

        pos_new = self.iterate()

        score_new = self._score(pos_new)
        self.evaluate(score_new)

        self.pos_l.append(pos_new)
        self.score_l.append(score_new)

        self.p_bar.update(score_new, pos_new, nth_iter)

    def _init_search(self):
        self.stop = False

        if "progress_bar" in self.verbosity:
            self.p_bar = ProgressBarLVL1(
                self.nth_process, self.n_iter, self.objective_function
            )
        else:
            self.p_bar = ProgressBarLVL0(
                self.nth_process, self.n_iter, self.objective_function
            )

        set_random_seed(self.nth_process, self.random_state)

    def _early_stop(self):
        if self.stop:
            return

        if self.max_time and time_exceeded(self.start_time, self.max_time):
            self.stop = True
        elif self.max_score and score_exceeded(self.p_bar.score_best, self.max_score):
            self.stop = True
        elif self.early_stopping and no_change(self.score_l, self.early_stopping):
            self.stop = True

    def print_info(self, *args):
        print_info(*args)

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
        random_state=None,
        nth_process=None,
    ):
        self.start_time = time.time()

        if verbosity is False:
            verbosity = []

        self.objective_function = objective_function
        self.n_iter = n_iter
        self.max_time = max_time
        self.max_score = max_score
        self.early_stopping = early_stopping
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.verbosity = verbosity
        self.random_state = random_state
        self.nth_process = nth_process

        self._init_search()

        # get init positions
        init = Initializer(self.conv)
        self.init_positions = init.set_pos(self.initialize)

        if isinstance(memory, DictProxy):
            mem = Memory(memory_warm_start, self.conv, dict_proxy=memory)
            self.score = self.results_mang.score(mem.memory(objective_function))
        elif memory is True:
            mem = Memory(memory_warm_start, self.conv)
            self.score = self.results_mang.score(mem.memory(objective_function))
        else:
            self.score = self.results_mang.score(objective_function)

        # loop to initialize N positions
        for init_pos, nth_iter in zip(self.init_positions, range(n_iter)):
            self._early_stop()
            if self.stop:
                break
            self._initialization(init_pos, nth_iter)

        self.finish_initialization()

        # loop to do the iterations
        for nth_iter in range(len(self.init_positions), n_iter):
            self._early_stop()
            if self.stop:
                break
            self._iteration(nth_iter)

        self.results = pd.DataFrame(self.results_mang.results_list)

        self.best_score = self.p_bar.score_best
        self.best_value = self.conv.position2value(self.p_bar.pos_best)
        self.best_para = self.conv.value2para(self.best_value)

        self.results["eval_time"] = self.eval_times
        self.results["iter_time"] = self.iter_times

        if memory is not False:
            self.memory_dict = mem.memory_dict
        else:
            self.memory_dict = {}

        self.p_bar.close()

        self.print_info(
            verbosity,
            self.objective_function,
            self.best_score,
            self.best_para,
            self.eval_times,
            self.iter_times,
            self.n_iter,
        )

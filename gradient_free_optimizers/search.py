# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random

import numpy as np
import pandas as pd

from .init_positions import Initializer
from .progress_bar import ProgressBarLVL0, ProgressBarLVL1
from .times_tracker import TimesTracker
from .memory import Memory
from .print_info import print_info


def time_exceeded(start_time, max_time):
    run_time = time.time() - start_time
    return max_time and run_time > max_time


def score_exceeded(score_best, max_score):
    return max_score and score_best >= max_score


def set_random_seed(nth_process, random_state):
    """
    Sets the random seed separately for each thread
    (to avoid getting the same results in each thread)
    """
    if nth_process is None:
        nth_process = 0

    if random_state is None:
        random_state = np.random.randint(0, high=2 ** 32 - 2)

    random.seed(random_state + nth_process)
    np.random.seed(random_state + nth_process)


class Search(TimesTracker):
    def __init__(self):
        super().__init__()

        self.optimizers = []
        self.new_results_list = []
        self.all_results_list = []

    @TimesTracker.eval_time
    def _score(self, pos):
        return self.score(pos)

    @TimesTracker.iter_time
    def _initialization(self, init_pos, nth_iter):
        self.init_pos(init_pos)

        score_new = self._score(init_pos)
        self.evaluate(score_new)

        self.p_bar.update(score_new, init_pos, nth_iter)

    @TimesTracker.iter_time
    def _iteration(self, nth_iter):
        pos_new = self.iterate()

        score_new = self._score(pos_new)
        self.evaluate(score_new)

        self.p_bar.update(score_new, pos_new, nth_iter)

    def _init_search(self):
        if "progress_bar" in self.verbosity:
            self.p_bar = ProgressBarLVL1(
                self.nth_process, self.n_iter, self.objective_function
            )
        else:
            self.p_bar = ProgressBarLVL0(
                self.nth_process, self.n_iter, self.objective_function
            )

        set_random_seed(self.nth_process, self.random_state)

        # get init positions
        init = Initializer(self.conv)
        init_positions = init.set_pos(self.initialize)

        return init_positions

    def _early_stop(self):
        if time_exceeded(self.start_time, self.max_time):
            return True
        elif score_exceeded(self.p_bar.score_best, self.max_score):
            return True
        else:
            return False

    def print_info(self, *args):
        print_info(*args)

    def search(
        self,
        objective_function,
        n_iter,
        max_time=None,
        max_score=None,
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
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.verbosity = verbosity
        self.random_state = random_state
        self.nth_process = nth_process

        init_positions = self._init_search()

        if memory is True:
            mem = Memory(memory_warm_start, self.conv)
            self.score = self.results_mang.score(mem.memory(objective_function))
        else:
            self.score = self.results_mang.score(objective_function)

        # loop to initialize N positions
        for init_pos, nth_iter in zip(init_positions, range(n_iter)):
            if self._early_stop():
                break
            self._initialization(init_pos, nth_iter)

        # loop to do the iterations
        for nth_iter in range(len(init_positions), n_iter):
            if self._early_stop():
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

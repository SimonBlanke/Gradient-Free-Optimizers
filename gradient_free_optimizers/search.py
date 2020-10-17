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
from .results_manager import ResultsManager
from .memory import Memory
from .converter import Converter
from .print_info import print_info

p_bar_dict = {
    False: ProgressBarLVL0,
    True: ProgressBarLVL1,
}


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

    def _init_memory(self, memory):
        memory_warm_start = self.memory_warm_start

        self.memory_dict = {}
        self.memory_dict_new = {}

        if isinstance(memory_warm_start, pd.DataFrame):
            parameter = set(self.search_space.keys())
            memory_para = set(memory_warm_start.columns)

            if parameter <= memory_para:
                values_list = list(
                    memory_warm_start[list(self.search_space.keys())].values
                )
                scores = memory_warm_start["score"]

                value_tuple_list = list(map(tuple, values_list))
                self.memory_dict = dict(zip(value_tuple_list, scores))
            else:
                missing = parameter - memory_para

                print(
                    "\nWarning:",
                    '"{}"'.format(*missing),
                    "is in search_space but not in memory dataframe",
                )
                print(
                    "Optimization run will continue "
                    "without memory warm start\n"
                )

    @TimesTracker.iter_time
    def _initialization(self, init_pos):
        self.init_pos(init_pos)

        score_new = self._score(init_pos)
        self.evaluate(score_new)

        self.p_bar.update(score_new, init_pos)

    @TimesTracker.iter_time
    def _iteration(self):
        pos_new = self.iterate()

        score_new = self._score(pos_new)
        self.evaluate(score_new)

        self.p_bar.update(score_new, pos_new)

    def _init_search(self):
        self._init_memory(self.memory)
        self.p_bar = p_bar_dict[self.progress_bar](
            self.nth_process, self.n_iter, self.objective_function
        )
        set_random_seed(self.nth_process, self.random_state)

        if self.warm_start is not None:
            self.initialize["warm_start"] = self.warm_start

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

    def search(
        self,
        objective_function,
        n_iter,
        initialize={"grid": 8, "random": 4, "vertices": 8},
        warm_start=None,
        max_time=None,
        max_score=None,
        memory=True,
        memory_warm_start=None,
        verbosity={
            "progress_bar": True,
            "print_results": True,
            "print_times": True,
        },
        random_state=None,
        nth_process=None,
    ):
        self.start_time = time.time()

        self.objective_function = objective_function
        self.n_iter = n_iter
        self.initialize = initialize
        self.warm_start = warm_start
        self.max_time = max_time
        self.max_score = max_score
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.progress_bar = verbosity["progress_bar"]
        self.random_state = random_state
        self.nth_process = nth_process

        conv = Converter(self.search_space)
        self.conv = conv
        results = ResultsManager(objective_function, conv)
        init_positions = self._init_search()

        if memory is True:
            mem = Memory(memory_warm_start, conv)
            self.score = mem.memory(results.score)
        else:
            self.score = results.score

        # loop to initialize N positions
        for init_pos, nth_iter in zip(init_positions, range(n_iter)):
            if self._early_stop():
                break
            self._initialization(init_pos)

        # loop to do the iterations
        for nth_iter in range(len(init_positions), n_iter):
            if self._early_stop():
                break
            self._iteration()

        self.results = pd.DataFrame(results.results_list)

        self.best_score = self.p_bar.score_best
        self.best_value = conv.position2value(self.p_bar.pos_best)
        self.best_para = conv.value2para(self.best_value)

        eval_time = np.array(self.eval_times).sum()
        iter_time = np.array(self.iter_times).sum()

        self.p_bar.close()

        print_info(
            self.objective_function,
            self.best_score,
            self.best_para,
            eval_time,
            iter_time,
            self.n_iter,
        )


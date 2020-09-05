# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random

import numpy as np
from tqdm import tqdm

from .init_positions import Initializer
from .progress_bar import ProgressBarLVL0, ProgressBarLVL1
from .conv import values2positions, position2value
from .times_tracker import TimesTracker

p_bar_dict = {
    False: ProgressBarLVL0,
    True: ProgressBarLVL1,
}


def time_exceeded(start_time, max_time):
    run_time = time.time() - start_time
    return max_time and run_time > max_time


def set_random_seed(nth_process, random_state):
    """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
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

    @TimesTracker.eval_time_dec
    def _score(self, pos):
        pos_tuple = tuple(pos)

        if self.memory is True and pos_tuple in self.memory_dict:
            return self.memory_dict[pos_tuple]
        else:
            score = self.objective_function(pos)
            self.memory_dict[pos_tuple] = score
            return score

    def _init_memory(self, memory):
        self.memory_dict = {}

        if isinstance(memory, dict):
            values_list = memory["values"]
            scores = memory["scores"]

            value_tuple_list = list(map(tuple, values_list))
            self.memory_dict = dict(zip(value_tuple_list, scores))
            self.memory = True

    @TimesTracker.iter_time_dec
    def _initialization(self, init_pos):
        self.init_pos(init_pos)

        value_new = position2value(self.search_space, init_pos)
        score_new = self._score(value_new)
        self.evaluate(score_new)

        self.p_bar.update(score_new, value_new)

    @TimesTracker.iter_time_dec
    def _iteration(self):
        pos_new = self.iterate()

        value_new = position2value(self.search_space, pos_new)
        score_new = self._score(value_new)
        self.evaluate(score_new)

        self.p_bar.update(score_new, value_new)

    def _init_search(self):
        self._init_memory(self.memory)
        self.p_bar = p_bar_dict[self.progress_bar](
            self.nth_process, self.n_iter, self.objective_function
        )
        set_random_seed(self.nth_process, self.random_state)

        # get init positions
        init = Initializer(self.search_space)
        init_positions = init.set_pos(self.initialize)

        return init_positions

    def search(
        self,
        objective_function,
        n_iter,
        initialize={"grid": 8, "random": 4, "vertices": 8},
        max_time=None,
        memory=True,
        progress_bar=True,
        print_results=True,
        random_state=None,
        nth_process=None,
    ):
        start_time = time.time()

        self.objective_function = objective_function
        self.n_iter = n_iter
        self.initialize = initialize
        self.memory = memory
        self.progress_bar = progress_bar
        self.random_state = random_state
        self.nth_process = nth_process

        init_positions = self._init_search()

        # loop to initialize N positions
        for init_pos in init_positions:
            if time_exceeded(start_time, max_time):
                break
            self._initialization(init_pos)

        # loop to do the iterations
        for nth_iter in range(len(init_positions), n_iter):
            if time_exceeded(start_time, max_time):
                break
            self._iteration()

        self.values = np.array(list(self.memory_dict.keys()))
        self.scores = np.array(list(self.memory_dict.values())).reshape(-1, 1)

        self.p_bar.close(print_results)

        self.best_score = self.p_bar.score_best
        self.best_values = self.p_bar.values_best


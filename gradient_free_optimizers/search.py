# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random

import numpy as np
from tqdm import tqdm

from .init_positions import Initializer
from .progress_bar import ProgressBarLVL0, ProgressBarLVL1
from .conv import values2positions, positions2values, position2value


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


class Search:
    def _score_mem(self, pos):
        pos_tuple = tuple(pos)

        if pos_tuple in self.memory_dict:
            return self.memory_dict[pos_tuple]
        else:
            score = self.objective_function(pos)
            self.memory_dict[pos_tuple] = score
            return score

    def _init_memory(self, memory):
        if memory == False:
            self._score = self.objective_function
        elif memory == True:
            self._score = self._score_mem
            self.memory_dict = {}
        elif isinstance(memory, dict):
            self._score = self._score_mem

            values_list = memory["values"]
            scores = memory["scores"]

            value_tuple_list = list(map(tuple, values_list))
            self.memory_dict = dict(zip(value_tuple_list, scores))

    def _initialization(self, init_pos):
        start_time_iter = time.time()
        self.init_pos(init_pos)

        value_new = position2value(self.search_space, init_pos)

        start_time_eval = time.time()
        score_new = self._score(value_new)
        self.p_bar.update(1, score_new, value_new)
        self.eval_times.append(time.time() - start_time_eval)

        self.evaluate(score_new)
        self.iter_times.append(time.time() - start_time_iter)

    def _iteration(self):
        start_time_iter = time.time()
        pos_new = self.iterate()

        value_new = position2value(self.search_space, pos_new)

        start_time_eval = time.time()
        score_new = self._score(value_new)
        self.p_bar.update(1, score_new, value_new)
        self.eval_times.append(time.time() - start_time_eval)

        self.evaluate(score_new)
        self.iter_times.append(time.time() - start_time_iter)

    def search(
        self,
        objective_function,
        n_iter,
        initialize={"grid": 0, "random": 0, "vertices": 5},
        max_time=None,
        memory=True,
        progress_bar=True,
        print_results=True,
        random_state=None,
        nth_process=None,
    ):
        start_time = time.time()

        self.objective_function = objective_function
        self._init_memory(memory)
        self.p_bar = p_bar_dict[progress_bar](nth_process, n_iter, objective_function)

        set_random_seed(nth_process, random_state)

        # get init positions
        init = Initializer(self.search_space)
        init_positions = init.set_pos(initialize)

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


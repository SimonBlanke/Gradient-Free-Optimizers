# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random

import numpy as np
from tqdm import tqdm

from .init_positions import init_grid_search, init_random_search
from .progress_bar import ProgressBarLVL0, ProgressBarLVL1


p_bar_dict = {
    False: ProgressBarLVL0,
    True: ProgressBarLVL1,
}


def time_exceeded(start_time, max_time):
    run_time = time.time() - start_time
    return max_time and run_time > max_time


def set_random_seed(nth_process, random_state):
    """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
    if random_state is None:
        random_state = np.random.randint(0, high=2 ** 32 - 2)

    random.seed(random_state + nth_process)
    np.random.seed(random_state + nth_process)


class Search:
    def _values2positions(self, values):
        init_pos_conv_list = []
        values_np = np.array(values)

        for n, space_dim in enumerate(self.search_space):
            pos_1d = values_np[:, n]
            init_pos_conv = np.where(space_dim == pos_1d)[0]
            init_pos_conv_list.append(init_pos_conv)

        return init_pos_conv_list

    def _positions2values(self, positions):
        pos_converted = []
        positions_np = np.array(positions)

        for n, space_dim in enumerate(self.search_space):
            pos_1d = positions_np[:, n]
            pos_conv = np.take(space_dim, pos_1d, axis=0)
            pos_converted.append(pos_conv)

        return list(np.array(pos_converted).T)

    def _init_values(self, init_values):
        init_positions_list = []

        if "random" in init_values:
            positions = init_random_search(self.space_dim, init_values["random"])
            init_positions_list.append(positions)
        if "grid" in init_values:
            positions = init_grid_search(self.space_dim, init_values["grid"])
            init_positions_list.append(positions)
        if "warm_start" in init_values:
            positions = self._values2positions(init_values["warm_start"])
            init_positions_list.append(positions)

        return [item for sublist in init_positions_list for item in sublist]

    def _position2value(self, position):
        value = []

        for n, space_dim in enumerate(self.search_space):
            value.append(space_dim[position[n]])

        return value

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

    def search(
        self,
        objective_function,
        n_iter,
        initialize={"grid": 7, "random": 3,},
        max_time=None,
        memory=True,
        progress_bar=True,
        print_results=True,
        random_state=None,
        nth_process=False,
    ):
        self.objective_function = objective_function
        self._init_memory(memory)

        set_random_seed(nth_process, random_state)
        start_time = time.time()

        self.p_bar = p_bar_dict[progress_bar](nth_process, n_iter, objective_function)

        init_values = self._init_values(initialize)

        # loop to initialize N positions
        for init_value in init_values:
            start_time_iter = time.time()
            self.init_pos(init_value)

            start_time_eval = time.time()
            score_new = self._score(init_value)
            self.p_bar.update(1, score_new, init_value)
            self.eval_times.append(time.time() - start_time_eval)

            self.evaluate(score_new)
            self.iter_times.append(time.time() - start_time_iter)

        # loop to do the iterations
        for nth_iter in range(len(init_values), n_iter):
            start_time_iter = time.time()
            pos_new = self.iterate()

            value_new = self._position2value(pos_new)

            start_time_eval = time.time()
            score_new = self._score(value_new)
            self.p_bar.update(1, score_new, value_new)
            self.eval_times.append(time.time() - start_time_eval)

            self.evaluate(score_new)
            self.iter_times.append(time.time() - start_time_iter)

            if time_exceeded(start_time, max_time):
                break

        self.values = np.array(list(self.memory_dict.keys()))
        self.scores = np.array(list(self.memory_dict.values())).reshape(-1, 1)

        self.p_bar.close()

        if print_results:
            print("\nResults: '{}'".format(objective_function.__name__), " ")
            print("  Best values", np.array(self.p_bar.values_best), " ")
            print("  Best score", self.p_bar.score_best, " ")


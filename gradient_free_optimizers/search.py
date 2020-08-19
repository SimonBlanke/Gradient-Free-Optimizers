# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
from tqdm import tqdm
import numpy as np

from .init_positions import init_grid_search, init_random_search
from .progress_bar import ProgressBarLVL0, ProgressBarLVL1
from .io_search_processor import init_search


p_bar_dict = {
    0: ProgressBarLVL0,
    1: ProgressBarLVL1,
}


def _time_exceeded(start_time, max_time):
    run_time = time.time() - start_time
    return max_time and run_time > max_time


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

    def _init_positions(self, init_values):
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

    def search(
        self,
        objective_function,
        n_iter,
        init_values={"grid": 7, "random": 3,},
        max_time=None,
        verbosity=1,
        random_state=None,
        nth_process=0,
    ):
        start_time = time.time()

        self.p_bar = p_bar_dict[verbosity]()
        self.p_bar.init(nth_process, n_iter, objective_function)

        init_positions = self._init_positions(init_values)

        # loop to initialize N positions
        for init_position in init_positions:
            start_time_iter = time.time()
            self.init_pos(init_position)

            start_time_eval = time.time()
            score_new = objective_function(init_position)
            self.p_bar.update(1, score_new)
            self.eval_times.append(time.time() - start_time_eval)

            self.evaluate(score_new)
            self.iter_times.append(time.time() - start_time_iter)

        # loop to do the iterations
        for nth_iter in range(len(init_positions), n_iter):
            start_time_iter = time.time()
            pos_new = self.iterate()

            start_time_eval = time.time()
            score_new = objective_function(pos_new)
            self.p_bar.update(1, score_new)
            self.eval_times.append(time.time() - start_time_eval)

            self.evaluate(score_new)
            self.iter_times.append(time.time() - start_time_iter)

            if _time_exceeded(start_time, max_time):
                break

        self.p_bar.close()


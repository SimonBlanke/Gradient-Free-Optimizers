# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from tqdm import tqdm


class ProgressBarBase:
    def __init__(self, nth_process, n_iter, objective_function):
        self.pos_best = None
        self._score_best = -np.inf
        self.score_best_list = []

        self.convergence_data = []

        self._best_since_iter = 0
        self.best_since_iter_list = []

        self.objective_function = objective_function

        self.n_iter_current = 0

    @property
    def score_best(self):
        return self._score_best

    @score_best.setter
    def score_best(self, score):
        self.score_best_list.append(score)
        self._score_best = score

    @property
    def best_since_iter(self):
        return self._best_since_iter

    @best_since_iter.setter
    def best_since_iter(self, nth_iter):
        self.best_since_iter_list.append(nth_iter)
        self._best_since_iter = nth_iter

    def _new2best(self, score_new, pos_new, nth_iter):
        if score_new > self.score_best:
            self.score_best = score_new
            self.pos_best = pos_new

        self.convergence_data.append(self.score_best)


class ProgressBarLVL0(ProgressBarBase):
    def __init__(self, nth_process, n_iter, objective_function):
        super().__init__(nth_process, n_iter, objective_function)

    def update(self, score_new, pos_new, nth_iter):
        self.n_iter_current = nth_iter
        self._new2best(score_new, pos_new, nth_iter)

    def close(self):
        pass


class ProgressBarLVL1(ProgressBarBase):
    def __init__(self, nth_process, n_iter, objective_function):
        super().__init__(nth_process, n_iter, objective_function)
        self._tqdm = tqdm(**self._tqdm_dict(nth_process, n_iter, objective_function))

    def update(self, score_new, pos_new, nth_iter):
        self.n_iter_current = nth_iter

        if score_new > self.score_best:
            self.best_since_iter = nth_iter

            self._tqdm.set_postfix(
                best_score=str(score_new),
                best_pos=str(pos_new),
                best_iter=str(self._best_since_iter),
            )

        self._new2best(score_new, pos_new, nth_iter)

        self._tqdm.update(1)
        # self._tqdm.refresh()

    def close(self):
        self._tqdm.close()

    def _tqdm_dict(self, nth_process, n_iter, objective_function):
        """
        Generates the parameter dict for tqdm in the iteration-loop of each optimizer
        """

        self.objective_function = objective_function

        if nth_process is None:
            process_str = ""
        else:
            process_str = str(objective_function.__name__)

        return {
            "total": n_iter,
            "desc": process_str,
            "position": nth_process,
            "leave": False,
            # "smoothing": 1.0,
        }

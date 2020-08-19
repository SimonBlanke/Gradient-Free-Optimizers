# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from tqdm.auto import tqdm


class ProgressBarLVL0:
    def __init__(self):
        pass

    def init(self, nth_process, n_iter, obj_func):
        pass

    def update(self, iter, score_new):
        pass

    def close(self):
        pass

    def _tqdm_dict(self, nth_process, n_iter, obj_func):
        pass


class ProgressBarLVL1:
    def __init__(self):
        self.best_since_iter = 0
        self.score_best = -np.inf
        # tqdm.set_lock(tqdm.get_lock())

    def init(self, nth_process, n_iter, obj_func):
        self._tqdm = tqdm(**self._tqdm_dict(nth_process, n_iter, obj_func))

    def update(self, iter, score_new):
        self._tqdm.update(iter)

        if score_new > self.score_best:
            self.score_best = score_new
            self.best_since_iter = self._tqdm.n
            self._tqdm.set_postfix(
                best_score=str(score_new), best_since_iter=self.best_since_iter
            )

    def close(self):
        self._tqdm.close()

    def _tqdm_dict(self, nth_process, n_iter, obj_func):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""
        return {
            "total": n_iter,
            "desc": "Process " + str(nth_process) + " -> " + obj_func.__name__,
            "position": nth_process,
            "leave": True,
        }


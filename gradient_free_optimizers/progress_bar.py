# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from tqdm.auto import tqdm


class ProgressBarLVL0:
    def __init__(self, nth_process, n_iter, obj_func):
        pass

    def update(self, iter, score_new):
        pass

    def close(self):
        pass

    def _tqdm_dict(self, nth_process, n_iter, obj_func):
        pass


class ProgressBarLVL1:
    def __init__(self, nth_process, n_iter, obj_func):
        self.best_since_iter = 0
        self.score_best = -np.inf
        self.values_best = None
        # tqdm.set_lock(tqdm.get_lock())

        self._tqdm = tqdm(**self._tqdm_dict(nth_process, n_iter, obj_func))

    def update(self, iter, score_new, values_new):
        self._tqdm.update(iter)

        if score_new > self.score_best:
            self.score_best = score_new
            self.values_best = values_new
            self.best_since_iter = self._tqdm.n
            self._tqdm.set_postfix(
                best_score=str(score_new), best_iter=str(self.best_since_iter)
            )

    def close(self):
        self._tqdm.close()

    def _tqdm_dict(self, nth_process, n_iter, obj_func):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""

        if nth_process is None:
            process_str = ""
        else:
            process_str = "Process " + str(nth_process) + " -> "

        return {
            "total": n_iter,
            "desc": process_str + obj_func.__name__,
            "position": nth_process,
            "leave": True,
        }


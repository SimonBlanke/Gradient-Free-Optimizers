# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from tqdm import tqdm


class ProgressBarBase:
    def __init__(self, nth_process, n_iter, objective_function):
        self.best_since_iter = 0
        self.score_best = -np.inf
        self.pos_best = None

        self.objective_function = objective_function

    def _new2best(self, score_new, pos_new):
        if score_new > self.score_best:
            self.score_best = score_new
            self.pos_best = pos_new


class ProgressBarLVL0(ProgressBarBase):
    def __init__(self, nth_process, n_iter, objective_function):
        super().__init__(nth_process, n_iter, objective_function)

    def update(self, score_new, pos_new):
        self._new2best(score_new, pos_new)

    def close(self):
        pass


class ProgressBarLVL1(ProgressBarBase):
    def __init__(self, nth_process, n_iter, objective_function):
        super().__init__(nth_process, n_iter, objective_function)
        self._tqdm = tqdm(
            **self._tqdm_dict(nth_process, n_iter, objective_function)
        )

    def update(self, score_new, pos_new):
        if score_new > self.score_best:
            self.best_since_iter = self._tqdm.n - 1
            self._tqdm.set_postfix(
                best_score=str(score_new),
                best_pos=str(pos_new),
                best_iter=str(self.best_since_iter),
            )

        self._new2best(score_new, pos_new)
        self._tqdm.update(1)
        # self._tqdm.refresh()

    def close(self):
        self._tqdm.close()

    def _tqdm_dict(self, nth_process, n_iter, objective_function):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""

        self.objective_function = objective_function

        if nth_process is None:
            process_str = ""
        else:
            process_str = "Process " + str(nth_process)

        return {
            "total": n_iter,
            "desc": process_str,
            "position": nth_process,
            "leave": False,
            # "smoothing": 1.0,
        }


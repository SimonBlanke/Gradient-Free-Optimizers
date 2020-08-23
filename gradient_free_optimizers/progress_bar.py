# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from tqdm import tqdm


class ProgressBarLVL0:
    def __init__(self, nth_process, n_iter, objective_function):
        pass

    def update(self, iter, score_new):
        pass

    def close(self, print_results):
        pass

    def _tqdm_dict(self, nth_process, n_iter, objective_function):
        pass


class ProgressBarLVL1:
    def __init__(self, nth_process, n_iter, objective_function):
        self.best_since_iter = 0
        self.score_best = -np.inf
        self.values_best = None

        self._tqdm = tqdm(**self._tqdm_dict(nth_process, n_iter, objective_function))

    def update(self, score_new, values_new):
        self._tqdm.update()
        self._tqdm.refresh()

        if score_new > self.score_best:
            self.score_best = score_new
            self.values_best = values_new
            self.best_since_iter = self._tqdm.n - 1
            self._tqdm.set_postfix(
                best_score=str(score_new), best_iter=str(self.best_since_iter)
            )

    def close(self, print_results):
        self._tqdm.close()

        if print_results:
            print("\nResults: '{}'".format(self.objective_function.__name__), " ")
            print("  Best values", np.array(self.values_best), " ")
            print("  Best score", self.score_best, " ")

    def _tqdm_dict(self, nth_process, n_iter, objective_function):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""

        self.objective_function = objective_function

        if nth_process is None:
            process_str = ""
        else:
            process_str = "Process " + str(nth_process) + " -> "

        return {
            "total": n_iter,
            "desc": process_str + objective_function.__name__,
            "position": nth_process,
            "leave": True,
            "smoothing": 1.0,
        }


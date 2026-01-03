# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SimpleProgressBar:
    """Fallback progress bar using print statements when tqdm is unavailable."""

    def __init__(self, total, desc="", position=None, leave=False):
        self.total = total
        self.desc = desc
        self.n = 0
        self.postfix = {}
        self._last_printed = -1

    def update(self, n=1):
        self.n += n
        percent = int(self.n / self.total * 100)
        if percent >= self._last_printed + 10 or self.n == self.total:
            self._last_printed = percent
            self._print_progress()

    def set_postfix(self, **kwargs):
        self.postfix = kwargs

    def _print_progress(self):
        bar_len = 20
        filled = int(bar_len * self.n / self.total)
        bar = "#" * filled + "-" * (bar_len - filled)

        postfix_str = ""
        if self.postfix:
            postfix_str = " | " + ", ".join(
                f"{k}={v}" for k, v in self.postfix.items()
            )

        desc_str = f"{self.desc}: " if self.desc else ""
        print(
            f"\r{desc_str}[{bar}] {self.n}/{self.total}{postfix_str}",
            end="",
            flush=True,
        )

        if self.n == self.total:
            print()

    def close(self):
        if self.n < self.total:
            print()


class ProgressBarBase:
    def __init__(self, nth_process, n_iter, objective_function):
        self.pos_best = None
        self._score_best = -math.inf
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
        progress_bar_class = tqdm if TQDM_AVAILABLE else SimpleProgressBar
        self._tqdm = progress_bar_class(
            **self._tqdm_dict(nth_process, n_iter, objective_function)
        )

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
        del self._tqdm

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

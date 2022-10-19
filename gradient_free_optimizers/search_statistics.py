# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class SearchStatistics:
    def __init__(self):
        super().__init__()

        self.nth_iter = 0

        self.n_init_total = 0
        self.n_iter_total = 0

    def init_stats(func):
        def wrapper(self, *args, **kwargs):
            self.n_init_search = 0
            self.n_iter_search = 0

            res = func(self, *args, **kwargs)
            return res

        return wrapper

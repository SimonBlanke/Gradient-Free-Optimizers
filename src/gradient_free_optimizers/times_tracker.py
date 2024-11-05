# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time


class TimesTracker:
    def __init__(self):
        super().__init__()

        self.eval_times = []
        self.iter_times = []

    def eval_time(func):
        def wrapper(self, *args, **kwargs):
            t = time.time()
            res = func(self, *args, **kwargs)
            self.eval_times.append(time.time() - t)
            return res

        return wrapper

    def iter_time(func):
        def wrapper(self, *args, **kwargs):
            t = time.time()
            res = func(self, *args, **kwargs)
            self.iter_times.append(time.time() - t)
            return res

        return wrapper


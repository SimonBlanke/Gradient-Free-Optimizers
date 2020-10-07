# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class Memory:
    def __init__(self, warm_start, conv):
        self.memory_dict = {}
        self.memory_dict_new = {}

        if warm_start is None:
            return

        if not isinstance(warm_start, pd.DataFrame):
            print("'memory_warm_start' must be of type pandas.DataFrame")
            return

        self.memory_dict = conv.dataframe2memory_dict(warm_start)

    def memory(self, score_func):
        def wrapper(pos):
            pos_tuple = tuple(pos)

            if pos_tuple in self.memory_dict:
                return self.memory_dict[pos_tuple]
            else:
                score = score_func(pos)

                self.memory_dict[pos_tuple] = score
                self.memory_dict_new[pos_tuple] = score

                return score

        return wrapper


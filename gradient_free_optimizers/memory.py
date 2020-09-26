# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class Memory:
    def __init__(self, warm_start, conv):
        self.memory_dict = {}
        self.memory_dict_new = {}

        if not isinstance(warm_start, pd.DataFrame):
            print("'memory_warm_start' must be of type pandas.DataFrame")
            return

        parameter = set(conv.search_space.keys())
        memory_para = set(warm_start.columns)

        if parameter <= memory_para:
            values = list(warm_start[list(conv.search_space.keys())].values)
            positions = conv.values2positions(values)
            scores = warm_start["score"]

            self.memory_dict = conv.positions_scores2memory_dict(positions, scores)
        else:
            missing = parameter - memory_para

            print(
                "\nWarning:",
                '"{}"'.format(*missing),
                "is in search_space but not in memory dataframe",
            )
            print("Optimization run will continue without memory warm start\n")

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


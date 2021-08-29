# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class Memory:
    def __init__(self, warm_start, conv, dict_proxy=None):
        self.memory_dict = {}
        self.memory_dict_new = {}
        self.conv = conv

        if dict_proxy is not None:
            self.memory_dict = dict_proxy

        if warm_start is None:
            return

        if not isinstance(warm_start, pd.DataFrame):
            print("Memory warm start must be of type pandas.DataFrame")
            print("Optimization will continue without memory warm start")

            return

        if len(warm_start) == 0:
            print("Memory warm start has no values in current search space")
            print("Optimization will continue without memory warm start")

            return

        self.memory_dict.update(self.conv.dataframe2memory_dict(warm_start))

    def memory(self, objective_function):
        def wrapper(para):
            value = self.conv.para2value(para)
            position = self.conv.value2position(value)

            pos_tuple = tuple(position)

            if pos_tuple in self.memory_dict:
                return self.memory_dict[pos_tuple]
            else:
                score = objective_function(para)

                self.memory_dict[pos_tuple] = score
                self.memory_dict_new[pos_tuple] = score

                return score

        return wrapper

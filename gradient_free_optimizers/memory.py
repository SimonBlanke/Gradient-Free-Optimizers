# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import logging
import pandas as pd

from multiprocessing.managers import DictProxy


class Memory:
    def __init__(self, warm_start, conv, memory=None):
        self.memory_dict = {}
        self.memory_dict_new = {}
        self.conv = conv

        if isinstance(memory, DictProxy):
            self.memory_dict = memory

        if warm_start is None:
            return

        if not isinstance(warm_start, pd.DataFrame):
            logging.warning("Memory warm start must be of type pandas.DataFrame")
            logging.warning("Optimization will continue without memory warm start")

            return

        if len(warm_start) == 0:
            logging.warning("Memory warm start has no values in current search space")
            logging.warning("Optimization will continue without memory warm start")

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

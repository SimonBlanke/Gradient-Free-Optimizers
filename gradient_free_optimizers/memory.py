# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd


class Memory:
    def __init__(self, warm_start, search_space):
        self.memory_dict = {}
        self.memory_dict_new = {}

        if isinstance(warm_start, pd.DataFrame):
            parameter = set(search_space.keys())
            memory_para = set(warm_start.columns)

            if parameter <= memory_para:
                values_list = list(warm_start[list(search_space.keys())].values)
                scores = warm_start["score"]

                value_tuple_list = list(map(tuple, values_list))
                self.memory_dict = dict(zip(value_tuple_list, scores))
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


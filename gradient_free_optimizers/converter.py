# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class Converter:
    def __init__(self, search_space):
        self.search_space = search_space
        self.para_names = list(search_space.keys())
        self.dim_sizes = np.array(
            [len(array) for array in search_space.values()]
        )
        self.search_space_positions = np.array(
            [range(len(array)) for array in search_space.values()]
        )
        self.max_positions = self.dim_sizes - 1
        self.search_space_values = list(search_space.values())

    def returnNoneIfArgNone(func):
        def wrapper(self, *args):
            for arg in [*args]:
                if arg is None:
                    return None
            else:
                return func(self, *args)

        return wrapper

    @returnNoneIfArgNone
    def position2value(self, position):
        value = []

        for n, space_dim in enumerate(self.search_space_values):
            value.append(space_dim[position[n]])

        return np.array(value)

    @returnNoneIfArgNone
    def value2position(self, value):
        position = []
        for n, space_dim in enumerate(self.search_space_values):
            pos = np.abs(value[n] - space_dim).argmin()
            position.append(pos)

        return np.array(position).astype(int)

    @returnNoneIfArgNone
    def value2para(self, value):
        para = {}
        for key, p_ in zip(self.para_names, value):
            para[key] = p_

        return para

    @returnNoneIfArgNone
    def para2value(self, para):

        value = []
        for para_name in self.para_names:
            value.append(para[para_name])

        return np.array(value)

    @returnNoneIfArgNone
    def values2positions(self, values):
        positions_temp = []
        values_np = np.array(values)

        for n, space_dim in enumerate(self.search_space_values):
            values_1d = values_np[:, n]
            m_conv = np.abs(values_1d - space_dim[:, np.newaxis])
            pos_list = m_conv.argmin(0)

            positions_temp.append(pos_list)

        positions = list(np.array(positions_temp).T.astype(int))

        return positions

    @returnNoneIfArgNone
    def positions2values(self, positions):
        values_temp = []
        positions_np = np.array(positions)

        for n, space_dim in enumerate(self.search_space_values):
            pos_1d = positions_np[:, n]
            value_ = np.take(space_dim, pos_1d, axis=0)
            values_temp.append(value_)

        values = list(np.array(values_temp).T)
        return values

    @returnNoneIfArgNone
    def positions_scores2memory_dict(self, positions, scores):
        value_tuple_list = list(map(tuple, positions))
        memory_dict = dict(zip(value_tuple_list, scores))

        return memory_dict

    @returnNoneIfArgNone
    def memory_dict2positions_scores(self, memory_dict):
        positions = [
            np.array(pos).astype(int) for pos in list(memory_dict.keys())
        ]
        scores = list(memory_dict.values())

        return positions, scores

    @returnNoneIfArgNone
    def dataframe2memory_dict(self, dataframe):
        parameter = set(self.search_space.keys())
        memory_para = set(dataframe.columns)

        if parameter <= memory_para:
            values = list(dataframe[self.para_names].values)
            positions = self.values2positions(values)
            scores = dataframe["score"]

            memory_dict = self.positions_scores2memory_dict(positions, scores)

            return memory_dict
        else:
            missing = parameter - memory_para

            print(
                "\nWarning:",
                '"{}"'.format(*missing),
                "is in search_space but not in memory dataframe",
            )
            print("Optimization run will continue without memory warm start\n")

            return {}

    @returnNoneIfArgNone
    def memory_dict2dataframe(self, memory_dict):
        positions, score = self.memory_dict2positions_scores(memory_dict)
        values = self.positions2values(positions)

        dataframe = pd.DataFrame(values, columns=self.para_names)
        dataframe["score"] = score

        return dataframe

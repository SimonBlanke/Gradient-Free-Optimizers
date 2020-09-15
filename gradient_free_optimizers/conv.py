# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


def position2value(search_space, position):
    value = []

    for n, space_dim in enumerate(search_space):
        value.append(space_dim[position[n]])

    return value


def value2position(search_space, value):
    position = []

    for n, space_dim in enumerate(search_space):
        pos = np.abs(value[n] - space_dim).argmin()
        position.append(pos)

    position = np.array(position)

    return position


"""
def values2positions(search_space, values):
    init_pos_conv_list = []
    values_np = np.array(values)

    print("\n values_np \n", values_np)
    print("\n search_space \n", search_space)

    for n, space_dim in enumerate(search_space):
        pos_1d = values_np[:, n]
        init_pos_conv = np.where(space_dim == pos_1d)[0]
        init_pos_conv_list.append(init_pos_conv)

    print("\n init_pos_conv_list \n", init_pos_conv_list)

    return init_pos_conv_list


def positions2values(search_space, positions):
    pos_converted = []
    positions_np = np.array(positions)

    for n, space_dim in enumerate(search_space):
        pos_1d = positions_np[:, n]
        pos_conv = np.take(space_dim, pos_1d, axis=0)
        pos_converted.append(pos_conv)

    return list(np.array(pos_converted).T)

"""


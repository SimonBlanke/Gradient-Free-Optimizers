# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


def init_grid_search(space_dim, n_pos):
    positions = []

    n_dim = len(space_dim)
    p_per_dim = int(np.power(n_pos, 1 / n_dim))

    for dim in space_dim:
        dim_dist = int(dim / (p_per_dim + 1))
        n_points = [n * dim_dist for n in range(1, p_per_dim + 1)]

        positions.append(n_points)

    pos_mesh = np.array(np.meshgrid(*positions))
    positions = list(pos_mesh.T.reshape(-1, n_dim))

    diff_pos = n_pos - len(positions)
    if diff_pos > 0:
        pos_rnd = init_random_search(space_dim, n_pos=diff_pos)

    return positions + pos_rnd


def init_random_search(space_dim, n_pos):
    positions = []
    for nth_pos in range(n_pos):
        pos = np.random.randint(space_dim, size=space_dim.shape)
        positions.append(pos)

    return positions

import numpy as np
import itertools


def split_into_subspaces(search_space, split_per_dim=2):
    sub_arrays = []
    for search_dim in search_space:
        sub_arrays.append(np.array_split(search_dim, split_per_dim))

    return [list(p) for p in itertools.product(*sub_arrays)]

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import math
import numpy as np


def get_search_space_size(search_space):
    size = 1
    for values in search_space.values():
        size *= len(values)
    return size


class SubSearchSpaces:
    def __init__(self, search_space, max_size, prioritize_dims=None):
        self.search_space = search_space
        self.max_size = max_size
        self.prioritize_dims = prioritize_dims

    def slice(self):
        sub_search_spaces = [self.search_space]
        sub_search_spaces_finished = []

        while sub_search_spaces:
            sub_search_spaces = self._get_sub_search_spaces(
                sub_search_spaces, self.max_size
            )

            (
                sub_search_spaces,
                sub_search_spaces_finished_tmp,
            ) = self._get_sub_search_spaces_finished(sub_search_spaces, self.max_size)
            sub_search_spaces_finished += sub_search_spaces_finished_tmp

        return sub_search_spaces_finished

    def _get_sub_search_spaces_finished(self, sub_search_spaces, max_size):
        sub_search_spaces_finished = []
        sub_search_spaces_new = []
        for sub_search_space in sub_search_spaces:
            search_space_size = get_search_space_size(sub_search_space)
            if search_space_size <= max_size:
                sub_search_spaces_finished.append(sub_search_space)
            else:
                sub_search_spaces_new.append(sub_search_space)

        return sub_search_spaces_new, sub_search_spaces_finished

    def _get_sub_search_spaces(self, sub_search_spaces, max_size):
        sub_search_spaces_new = []
        for sub_search_space in sub_search_spaces:
            sub_search_space_new = self._slice_search_space(sub_search_space, max_size)
            sub_search_spaces_new += sub_search_space_new
        return sub_search_spaces_new

    def _slice_search_space(self, search_space, max_size):
        search_space_size = get_search_space_size(search_space)

        if search_space_size <= max_size:
            return [search_space]

        paras = list(search_space.keys())

        space_dims = {}
        for para in paras:
            dim = len(list(search_space[para]))
            if dim == 1:
                continue
            space_dims[para] = dim

        if self.prioritize_dims is None:
            para_sort_size = sorted(space_dims, key=space_dims.get)
            para_slice = para_sort_size[0]
        else:
            para_sort_prio = reversed(
                sorted(self.prioritize_dims, key=self.prioritize_dims.get)
            )

            for para_prio in para_sort_prio:
                if len(search_space[para_prio]) != 1:
                    para_slice = para_prio
                    break

        return self._slice_dim(search_space, para_slice, max_size)

    def _slice_dim(self, search_space, para_slice, max_size):
        search_space_size = get_search_space_size(search_space)

        paras = list(search_space.keys())
        space_dims = {}
        for para in paras:
            dim = len(list(search_space[para]))
            if dim == 1:
                continue
            space_dims[para] = dim

        smallest_dim = search_space[para_slice]
        smallest_dim_size = space_dims[para_slice]

        if search_space_size / smallest_dim_size <= max_size:
            n_regions = min(
                int(math.ceil(search_space_size / max_size)), smallest_dim_size
            )
        else:
            n_regions = smallest_dim_size

        sub_regions = np.array_split(smallest_dim, n_regions)
        sub_search_spaces = []
        for sub_region in sub_regions:
            sub_search_space = {}

            for para in paras:
                if para == para_slice:
                    sub_search_space[para] = sub_region
                else:
                    sub_search_space[para] = search_space[para]

            sub_search_spaces.append(sub_search_space)
        return sub_search_spaces

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
from scipy.spatial.distance import cdist

from ..smb_opt.smbo import SMBO
from ...search import Search


class SubSpace:
    def __init__(self, search_space):
        self.search_space = search_space

        self.score = None
        self.center_pos = self.center_pos_()
        self.biggest_dim = self.biggest_dim_()

    def center_pos_(self):
        center_pos = []

        for dim in list(self.search_space.keys()):
            dim_array = self.search_space[dim]
            array_size = dim_array.shape[0]
            center_idx = int(array_size / 2)

            center_pos.append(dim_array[center_idx])

        return np.array(center_pos).astype(int)

    def biggest_dim_(self):
        largest_dim = None
        largest_size = 0

        for dim in list(self.search_space.keys()):
            dim_array = self.search_space[dim]
            array_size = dim_array.shape[0]

            if array_size == largest_size:
                if random.randint(0, 1):
                    largest_size = array_size
                    largest_dim = dim

            elif array_size > largest_size:
                largest_size = array_size
                largest_dim = dim

        return largest_dim

    def lipschitz_bound_(self, score, K=1):
        self.score = score

        furthest_pos_ = []

        for dim in list(self.search_space.keys()):
            dim_array = self.search_space[dim]
            furthest_pos_.append(dim_array[0])
        furthest_pos = np.array(furthest_pos_)

        dist = cdist(furthest_pos.reshape(1, -1), self.center_pos.reshape(1, -1))

        self.lipschitz_bound = score + K * dist


class DirectAlgorithm(SMBO, Search):
    name = "Direct Algorithm"
    _name_ = "direct_algorithm"
    __name__ = "DirectAlgorithm"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subspace_l = []

    def select_next_subspace(self):
        for subspace in self.subspace_l:
            if subspace.score is None:
                return subspace

    def split_dim_into_n(self, subspace, n_splits=3):
        search_space = subspace.search_space
        dim_array = search_space[subspace.biggest_dim]

        sub_arrays = np.array_split(dim_array, n_splits)

        sub_search_space_l = []
        for sub_array in sub_arrays:
            sub_search_space_ = dict(search_space)
            sub_search_space_[subspace.biggest_dim] = sub_array

            sub_search_space_l.append(sub_search_space_)

        for search_space_ in sub_search_space_l:
            try:
                self.subspace_l.append(SubSpace(search_space_))
            except IndexError:
                pass

        self.subspace_l.remove(subspace)

    def select_subspace(self):
        lipschitz_bound_max = -np.inf
        next_subspace = None

        for subspace in self.subspace_l:
            if subspace.lipschitz_bound > lipschitz_bound_max:
                lipschitz_bound_max = subspace.lipschitz_bound
                next_subspace = subspace

        # if lipschitz_bound is nan or -inf
        if next_subspace is None:
            next_subspace = self.subspace_l[0]

        return next_subspace

    def finish_initialization(self):
        subspace = SubSpace(self.conv.pos_space)
        self.subspace_l.append(subspace)
        self.search_state = "iter"

    @SMBO.track_new_pos
    @SMBO.track_X_sample
    def iterate(self):
        self.current_subspace = self.select_next_subspace()
        if self.current_subspace:
            return self.current_subspace.center_pos

        else:
            self.current_subspace = self.select_subspace()
            self.split_dim_into_n(self.current_subspace)

            return self.subspace_l[-1].center_pos

    @SMBO.track_new_score
    def evaluate(self, score_new):
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.pos_current = self.pos_new

            self.score_best = score_new
            self.score_current = score_new

        if self.search_state == "iter":
            self.current_subspace.lipschitz_bound_(score_new)

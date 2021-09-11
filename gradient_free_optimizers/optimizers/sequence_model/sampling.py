# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
import random
import numpy as np


class InitialSampler:
    def __init__(self, conv, max_sample_size, dim_max_sample_size=1000000):
        self.conv = conv
        self.max_sample_size = max_sample_size
        self.dim_max_sample_size = dim_max_sample_size

    def get_pos_space(self):
        if self.max_sample_size < self.conv.search_space_size:
            n_samples_array = self.get_n_samples_dims()
            return self.random_choices(n_samples_array)
        else:
            if self.conv.max_dim < 255:
                _dtype = np.uint8
            elif self.conv.max_dim < 65535:
                _dtype = np.uint16
            elif self.conv.max_dim < 4294967295:
                _dtype = np.uint32
            else:
                _dtype = np.uint64

            pos_space = []
            for dim_ in self.conv.dim_sizes:
                pos_space.append(np.arange(dim_, dtype=_dtype))

            return pos_space

    def get_n_samples_dims(self):
        # TODO of search space is > 33 dims termination criterion must be:
        # "search_space_size < self.max_sample_size"

        dim_sizes_temp = self.conv.dim_sizes
        dim_sizes_temp = np.clip(
            dim_sizes_temp, a_min=1, a_max=self.dim_max_sample_size
        )
        search_space_size = self.conv.dim_sizes.prod()

        while abs(search_space_size - self.max_sample_size) > self.max_sample_size / 10:
            n_samples_array = []
            for idx, dim_size in enumerate(np.nditer(dim_sizes_temp)):
                array_diff_ = random.randint(1, dim_size)
                n_samples_array.append(array_diff_)

                sub = int((dim_size / 1000) ** 1.5)
                dim_sizes_temp[idx] = np.maximum(1, dim_size - sub)

            search_space_size = np.array(n_samples_array).prod()

        return n_samples_array

    def random_choices(self, n_samples_array):
        pos_space = []
        for n_samples, dim_size in zip(n_samples_array, self.conv.dim_sizes):

            if dim_size > self.dim_max_sample_size:
                pos_space.append(
                    np.random.randint(low=1, high=dim_size, size=n_samples)
                )
            else:
                pos_space.append(
                    np.random.choice(dim_size, size=n_samples, replace=False)
                )

        return pos_space

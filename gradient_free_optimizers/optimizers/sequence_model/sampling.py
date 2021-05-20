# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
import random
import numpy as np


class InitialSampler:
    def __init__(self, conv, init_sample_size, dim_max_sample_size=1000000):
        self.conv = conv
        self.init_sample_size = init_sample_size
        self.dim_max_sample_size = dim_max_sample_size

    def get_pos_space(self):
        print("\nself.init_sample_size", self.init_sample_size)
        print("self.conv.dim_sizes", self.conv.dim_sizes)
        print("self.conv.search_space_size", self.conv.search_space_size)
        print("\n")

        if self.init_sample_size < self.conv.search_space_size:
            n_samples_array = self.get_n_samples_dims()
            return self.random_choices(n_samples_array)
        else:
            print("\n no sampling! \n")
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
        dim_sizes_temp = self.conv.dim_sizes
        dim_sizes_temp = np.clip(
            dim_sizes_temp, a_min=1, a_max=self.dim_max_sample_size
        )

        search_space_size = self.conv.dim_sizes.prod()

        print("1")
        while abs(search_space_size - self.init_sample_size) > 10000:
            n_samples_array = []
            for idx, dim_size in enumerate(np.nditer(dim_sizes_temp)):
                array_diff_ = random.randint(1, dim_size)
                n_samples_array.append(array_diff_)

                sub = int((dim_size / 1000) ** 1.5)
                dim_sizes_temp[idx] = dim_size - sub

            # array_diff = np.array(array_diff)
            # n_samples_array = np.max(1, np.subtract(dim_sizes_temp, array_diff))

            search_space_size = np.array(n_samples_array).prod()

        print("\n\nsearch_space_size", search_space_size)
        print("n_samples_array", n_samples_array)

        return n_samples_array

    def random_choices(self, n_samples_array):
        print("2")

        pos_space = []
        for n_samples, dim_size in zip(n_samples_array, self.conv.dim_sizes):
            print("\n  n_samples", n_samples)

            if dim_size > self.dim_max_sample_size:
                pos_space.append(
                    np.random.randint(low=1, high=dim_size, size=n_samples)
                )
            else:
                pos_space.append(
                    np.random.choice(dim_size, size=n_samples, replace=False)
                )
            print("  rnd finished")
        print("3")

        return pos_space

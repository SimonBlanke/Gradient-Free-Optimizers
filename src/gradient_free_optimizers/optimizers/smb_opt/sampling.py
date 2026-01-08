# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
import random

from gradient_free_optimizers._array_backend import (
    array,
    arange,
    clip,
    maximum,
    prod,
    inf,
    random as np_random,
)

# Maximum supported dimensions for SMBO sampling
# Limited by memory requirements for full position space enumeration
MAX_SUPPORTED_DIMENSIONS = 31


class InitialSampler:
    def __init__(self, conv, max_sample_size, dim_max_sample_size=1000000):
        self.conv = conv
        self.max_sample_size = max_sample_size
        self.dim_max_sample_size = dim_max_sample_size

        if self.conv.n_dimensions > MAX_SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"Search space has {self.conv.n_dimensions} dimensions, but "
                f"sequential model-based optimizers support at most "
                f"{MAX_SUPPORTED_DIMENSIONS} dimensions due to memory constraints. "
                f"Consider using a population-based optimizer (e.g., "
                f"EvolutionStrategyOptimizer, ParticleSwarmOptimizer) for "
                f"high-dimensional problems."
            )

    def get_pos_space(self):
        if self.max_sample_size < self.conv.search_space_size:
            n_samples_array = self.get_n_samples_dims()
            return self.random_choices(n_samples_array)
        else:
            pos_space = []
            for dim_ in self.conv.dim_sizes:
                pos_space.append(arange(int(dim_)))

            return pos_space

    def get_n_samples_dims(self):
        dim_sizes_temp = array(list(self.conv.dim_sizes))
        dim_sizes_temp = clip(dim_sizes_temp, 1, self.dim_max_sample_size)
        search_space_size = prod(self.conv.dim_sizes)
        if search_space_size == 0:
            search_space_size = inf

        while abs(search_space_size) > self.max_sample_size:
            n_samples_array = []
            for idx in range(len(dim_sizes_temp)):
                dim_size = int(dim_sizes_temp[idx])
                array_diff_ = random.randint(1, dim_size)
                n_samples_array.append(array_diff_)

                sub = max(int(dim_size - (dim_size**0.999)), 1)
                dim_sizes_temp[idx] = max(1, dim_size - sub)

            search_space_size = prod(array(n_samples_array))
            if search_space_size == 0:
                search_space_size = inf

        return n_samples_array

    def random_choices(self, n_samples_array):
        pos_space = []
        for n_samples, dim_size in zip(n_samples_array, self.conv.dim_sizes):
            dim_size = int(dim_size)
            if dim_size > self.dim_max_sample_size:
                pos_space.append(
                    np_random.randint(low=1, high=dim_size, size=n_samples)
                )
            else:
                pos_space.append(
                    np_random.choice(dim_size, size=n_samples, replace=False)
                )

        return pos_space

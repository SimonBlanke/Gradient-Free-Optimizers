# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Sampling strategies for sequential model-based optimizers."""

import random

from gradient_free_optimizers._array_backend import (
    arange,
    array,
    clip,
    inf,
    prod,
)
from gradient_free_optimizers._array_backend import (
    random as np_random,
)
from gradient_free_optimizers._dimension_types import DimensionType

# Maximum supported dimensions for SMBO sampling
# Limited by memory requirements for full position space enumeration
MAX_SUPPORTED_DIMENSIONS = 31


class InitialSampler:
    """Sample positions from search space for sequential model-based optimizers."""

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
        """Generate position space for SMBO sampling.

        For discrete and categorical dimensions: generates index arrays.
        For continuous dimensions: samples floats uniformly from the range.
        """
        if self.max_sample_size < self.conv.search_space_size:
            n_samples_array = self.get_n_samples_dims()
            return self.random_choices(n_samples_array)
        else:
            pos_space = []
            for idx, dim_size in enumerate(self.conv.dim_sizes):
                dim_type = self.conv.dim_types[idx]

                if dim_type == DimensionType.CONTINUOUS:
                    # Sample floats uniformly in the continuous range
                    min_val, max_val = self.conv.dim_infos[idx].bounds
                    # Use a reasonable number of samples for continuous dims
                    n_samples = min(1000, self.max_sample_size)
                    samples = np_random.uniform(min_val, max_val, n_samples)
                    pos_space.append(samples)
                else:
                    # Discrete and categorical: use index-based sampling
                    pos_space.append(arange(int(dim_size)))

            return pos_space

    def get_n_samples_dims(self):
        """Compute number of samples for each dimension.

        For continuous dimensions, uses a fixed sample count since
        dim_sizes is 1 (placeholder) for continuous.
        """
        # Default samples for continuous dimensions
        continuous_samples = min(100, self.max_sample_size)

        # Build effective dim sizes (replacing placeholder 1 for continuous)
        effective_sizes = []
        for idx, dim_size in enumerate(self.conv.dim_sizes):
            dim_type = self.conv.dim_types[idx]
            if dim_type == DimensionType.CONTINUOUS:
                effective_sizes.append(continuous_samples)
            else:
                effective_sizes.append(dim_size)

        dim_sizes_temp = array(effective_sizes)
        dim_sizes_temp = clip(dim_sizes_temp, 1, self.dim_max_sample_size)
        search_space_size = prod(dim_sizes_temp)
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
        """Generate random samples for each dimension.

        For discrete and categorical: random indices.
        For continuous: random floats in the range.
        """
        pos_space = []
        for idx, (n_samples, dim_size) in enumerate(
            zip(n_samples_array, self.conv.dim_sizes)
        ):
            dim_type = self.conv.dim_types[idx]

            if dim_type == DimensionType.CONTINUOUS:
                # Sample floats uniformly in the continuous range
                min_val, max_val = self.conv.dim_infos[idx].bounds
                pos_space.append(np_random.uniform(min_val, max_val, n_samples))
            else:
                # Discrete and categorical: sample indices
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

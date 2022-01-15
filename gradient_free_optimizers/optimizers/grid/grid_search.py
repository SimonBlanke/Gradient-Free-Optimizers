# Author: ...
# Email: ...
# License: MIT License

import numpy as np

try:
    from fractions import gcd
except:
    from math import gcd

from ..base_optimizer import BaseOptimizer
from ...search import Search


class GridSearchOptimizer(BaseOptimizer, Search):
    def __init__(self, *args, step_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_position = np.zeros((self.conv.n_dimensions,), dtype=int)
        # current position mapped in [|0,search_space_size-1|] (1D)
        self.high_dim_pointer = 0
        # direction is a generator of our search space (prime with search_space_size)
        self.direction = None
        # step_size describes how many steps of size direction are jumped at each iteration
        self.step_size = step_size

    def get_direction(self):
        """
        Aim here is to generate a prime number of the search space size we call our direction.
        As direction is prime with the search space size, we know it is
        a generator of Z/(search_space_size*Z).
        """
        n_dims = self.conv.n_dimensions
        search_space_size = self.conv.search_space_size

        # Find prime number near search_space_size ** (1/n_dims)
        dim_root = int(np.round(np.power(search_space_size, 1 / n_dims)))
        is_prime = False

        while not is_prime:
            if gcd(int(search_space_size), int(dim_root)) == 1:
                is_prime = True
            else:
                dim_root += -1

        return dim_root

    def grid_move(self):
        """
        We convert our pointer of Z/(search_space_size * Z) in a position of our search space.
        This algorithm uses a bijection from Z/(search_space_size * Z) -> (Z/dim_1*Z)x...x(Z/dim_n*Z).
        """
        new_pos = []
        dim_sizes = self.conv.dim_sizes
        pointer = self.high_dim_pointer

        # The coordinate of our new position for each dimension is
        # the quotient of the pointer by the product of remaining dimensions
        # Describes a bijection from Z/search_space_size*Z -> (Z/dim_1*Z)x...x(Z/dim_n*Z)
        for dim in range(len(dim_sizes) - 1):
            new_pos.append(pointer // np.prod(dim_sizes[dim + 1 :]) % dim_sizes[dim])
            pointer = pointer % np.prod(dim_sizes[dim + 1 :])
        new_pos.append(pointer)
        return np.array(new_pos)

    @BaseOptimizer.track_nth_iter
    def iterate(self):
        # If this is the first iteration:
        # Generate the direction and return initial_position
        if self.direction is None:
            self.direction = self.get_direction()
            return self.initial_position

        # If this is not the first iteration:
        # Update high_dim_pointer by taking a step of size step_size * direction.

        # Multiple passes are needed in order to observe the entire search space
        # depending on the step_size parameter.
        _, current_pass = self.nth_iter, self.high_dim_pointer % self.step_size
        current_pass_finished = (
            (self.nth_iter + 1) * self.step_size // self.conv.search_space_size
            > self.nth_iter * self.step_size // self.conv.search_space_size
        )
        # Begin the next pass if current is finished.
        if current_pass_finished:
            self.high_dim_pointer = current_pass + 1
        else:
            # Otherwise update pointer in Z/(search_space_size*Z)
            # using the prime step direction and step_size.
            self.high_dim_pointer = (
                self.high_dim_pointer + self.step_size * self.direction
            ) % self.conv.search_space_size

        # Compute corresponding position in our search space.

        pos_new = self.grid_move()
        pos_new = self.conv2pos(pos_new)
        return pos_new

# Author: ...
# Email: ...
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer
from ...search import Search

class GridSearchOptimizer(BaseOptimizer, Search):
    def __init__(
        self,
        search_space,
        initialize={},
    ):
        super().__init__(search_space, initialize)
        self.initial_position = np.zeros((self.conv.n_dimensions,), dtype=int)
        self.high_dim_pointer = 0
        self.direction = None
        
        
    def get_direction(self):
        """
        Aim here is to generate a prime number of the search space size we call our direction.
        As direction is prime with the search space size, we know it is a generator of Z/(search_space_size*Z).
        """
        n_dims = self.conv.n_dimensions
        search_space_size = self.conv.search_space_size
        
        # Find prime number near search_space_size ** (1/n_dims)
        dim_root = int(np.round(np.power(search_space_size,1/n_dims)))
        is_prime = False

        while not is_prime:
            if np.gcd(int(search_space_size),int(dim_root)) == 1: 
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
        for dim in range(len(dim_sizes)-1):
            new_pos.append(pointer // np.prod(dim_sizes[dim+1:]))
            pointer = pointer % np.prod(dim_sizes[dim+1:])
        new_pos.append(pointer)
        return np.array(new_pos)

    
    @BaseOptimizer.track_nth_iter
    def iterate(self):
        if self.direction is None:
            self.direction = self.get_direction()
            return self.initial_position
        else:
            # Update pointer in Z/(search_space_size*Z) using the prime step self.direction
            self.high_dim_pointer = (self.high_dim_pointer + self.direction) % self.conv.search_space_size
            # Compute corresponding position in our search space
            return self.grid_move()
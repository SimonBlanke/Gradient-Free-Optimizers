# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from math import gcd

from gradient_free_optimizers._array_backend import array, power, prod, zeros

from ..base_optimizer import BaseOptimizer


class DiagonalGridSearchOptimizer(BaseOptimizer):
    """Diagonal grid search using number theory for space coverage.

    Uses a prime generator to traverse the search space diagonally,
    ensuring diverse coverage early in the search. The traversal visits
    positions that are spread across multiple dimensions simultaneously.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default={"grid": 4, "random": 2, "vertices": 4}
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    step_size : int, default=1
        Step multiplier for grid traversal.
    """

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        step_size=1,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.initial_position = zeros((self.conv.n_dimensions,)).astype(int)
        # current position mapped in [|0,search_space_size-1|] (1D)
        self.high_dim_pointer = 0
        # direction is a generator of our search space (prime with search_space_size)
        self.direction_calc = None
        # step_size: how many steps of size direction are jumped each iteration
        self.step_size = step_size

    def get_direction(self):
        """Generate a prime number to serve as direction in search space.

        As direction is prime with the search space size, we know it is
        a generator of Z/(search_space_size*Z).
        """
        n_dims = self.conv.n_dimensions
        search_space_size = self.conv.search_space_size

        # Find prime number near search_space_size ** (1/n_dims)
        dim_root = int(round(power(search_space_size, 1 / n_dims)))
        is_prime = False

        while not is_prime:
            if gcd(int(search_space_size), int(dim_root)) == 1:
                is_prime = True
            else:
                dim_root += -1

        return dim_root

    def grid_move(self):
        """Convert 1D pointer to a position in the multi-dimensional search space.

        Uses a bijection from Z/(search_space_size * Z) to the product space.
        """
        new_pos = []
        dim_sizes = self.conv.dim_sizes
        pointer = self.high_dim_pointer

        # The coordinate of our new position for each dimension is
        # the quotient of the pointer by the product of remaining dimensions.
        # Bijection: Z/search_space_size*Z -> (Z/dim_1*Z)x...x(Z/dim_n*Z)
        for dim in range(len(dim_sizes) - 1):
            remaining_prod = prod(dim_sizes[dim + 1 :])
            new_pos.append(pointer // remaining_prod % dim_sizes[dim])
            pointer = pointer % remaining_prod
        new_pos.append(pointer)

        return array(new_pos)

    @BaseOptimizer.track_new_pos
    def iterate(self):
        """Generate next diagonal grid position."""
        # while loop for constraint opt
        while True:
            # If this is the first iteration:
            # Generate the direction and return initial_position
            if self.direction_calc is None:
                self.direction_calc = self.get_direction()

                pos_new = self.initial_position
                if self.conv.not_in_constraint(pos_new):
                    return pos_new
                else:
                    return self.move_random()

            # If this is not the first iteration:
            # Update high_dim_pointer by taking a step of size step_size * direction.

            # Multiple passes are needed in order to observe the entire search space
            # depending on the step_size parameter.
            _, current_pass = (
                self.nth_trial,
                self.high_dim_pointer % self.step_size,
            )
            current_pass_finished = (
                (self.nth_trial + 1) * self.step_size // self.conv.search_space_size
                > self.nth_trial * self.step_size // self.conv.search_space_size
            )
            # Begin the next pass if current is finished.
            if current_pass_finished:
                self.high_dim_pointer = current_pass + 1
            else:
                # Otherwise update pointer in Z/(search_space_size*Z)
                # using the prime step direction and step_size.
                self.high_dim_pointer = (
                    self.high_dim_pointer + self.step_size * self.direction_calc
                ) % self.conv.search_space_size

            # Compute corresponding position in our search space.

            pos_new = self.grid_move()
            pos_new = self.conv2pos(pos_new)

            if self.conv.not_in_constraint(pos_new):
                return pos_new

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random

from gradient_free_optimizers._array_backend import array
from gradient_free_optimizers._array_backend import random as np_random


def set_random_seed(nth_process, random_state):
    """Set random seed separately for each thread to avoid duplicate results."""
    if nth_process is None:
        nth_process = 0

    if random_state is None:
        random_state = random.randint(0, 2**31 - 2)

    random.seed(random_state + nth_process)
    np_random.seed(random_state + nth_process)

    return random_state + nth_process


def move_random(ss_positions):
    """Select a random element from each sublist and return as array.

    Parameters
    ----------
    ss_positions : list
        A list of lists representing search spaces of possible positions.

    Returns
    -------
    array
        One randomly selected element from each sublist.
    """
    return array([random.choice(search_space_pos) for search_space_pos in ss_positions])

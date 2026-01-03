# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random

from gradient_free_optimizers._array_backend import array, random as np_random


def set_random_seed(nth_process, random_state):
    """
    Sets the random seed separately for each thread
    (to avoid getting the same results in each thread)
    """
    if nth_process is None:
        nth_process = 0

    if random_state is None:
        random_state = random.randint(0, 2**31 - 2)

    random.seed(random_state + nth_process)
    np_random.seed(random_state + nth_process)

    return random_state + nth_process


def move_random(ss_positions):
    """
    Selects a random element from each sublist in ss_positions and returns them as an array.

    Args:
    ss_positions (list): A list of lists representing search spaces of possible positions.

    Returns:
    array: Array containing one randomly selected element from each sublist in ss_positions.
    """
    return array([random.choice(search_space_pos) for search_space_pos in ss_positions])

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


def set_random_seed(nth_process, random_state):
    """
    Sets the random seed separately for each thread
    (to avoid getting the same results in each thread)
    """
    if nth_process is None:
        nth_process = 0

    if random_state is None:
        random_state = np.random.randint(
            0, high=2**31 - 2, dtype=np.int64
        ).item()

    random.seed(random_state + nth_process)
    np.random.seed(random_state + nth_process)

    return random_state + nth_process


def move_random(ss_positions):
    """
    Selects a random element from each sublist in ss_positions and returns them as a NumPy array.

    Args:
    ss_positions (list): A list of lists representing search spaces of possible positions.

    Returns:
    np.array: NumPy array containing one randomly selected element from each sublist in ss_positions.
    """
    return np.array(
        [random.choice(search_space_pos) for search_space_pos in ss_positions]
    )

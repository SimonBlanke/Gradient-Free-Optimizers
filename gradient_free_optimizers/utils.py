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
        random_state = np.random.randint(0, high=2 ** 31 - 2, dtype=np.int64)

    random.seed(random_state + nth_process)
    np.random.seed(random_state + nth_process)


def move_random(ss_positions):
    position = []
    for search_space_pos in ss_positions:
        pos_ = random.choice(search_space_pos)
        position.append(pos_)

    return np.array(position)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np
import logging
from typing import List, Optional, Tuple, Set, Union, Any

# Configure logger
logger = logging.getLogger(__name__)

def set_random_seed(nth_process: Optional[int], random_state: Optional[int]) -> int:
    """
    Sets the random seed separately for each process to ensure reproducibility
    while avoiding identical results across processes.
    
    Parameters
    ----------
    nth_process : int, optional
        Process identifier used to create a unique seed for each process.
        If None, defaults to 0.
    random_state : int, optional
        Base random seed. If None, a random value is generated.
        
    Returns
    -------
    int
        The final random seed used (random_state + nth_process)
    """
    if nth_process is None:
        nth_process = 0

    if random_state is None:
        random_state = np.random.randint(
            0, high=2**31 - 2, dtype=np.int64
        ).item()

    # Avoid potential integer overflow
    final_seed = (random_state + nth_process) % (2**31 - 1)
    
    random.seed(final_seed)
    np.random.seed(final_seed)

    return final_seed


def move_random(ss_positions: List[List[Any]]) -> np.ndarray:
    """
    Selects a random element from each sublist in ss_positions.
    
    This function is used across the codebase as a basic building block for 
    random position generation in the search space.

    Parameters
    ----------
    ss_positions : list of lists
        A list of lists representing search spaces of possible positions,
        where each sublist represents possible values for one dimension.

    Returns
    -------
    np.ndarray
        NumPy array containing one randomly selected element from each sublist.
    """
    return np.array(
        [random.choice(search_space_pos) for search_space_pos in ss_positions]
    )


class RandomSearchOptimizer:
    """
    Random Search optimizer supporting sampling with or without replacement.
    
    This class is used by various optimization algorithms in the codebase
    to generate random positions within the search space.

    Parameters
    ----------
    ss_positions : list of lists
        Lists representing search spaces per dimension.
    replacement : bool, default=True
        If True, allows sampling the same position multiple times.
        If False, ensures each sampled position is unique.
    max_attempts : int, default=1000
        Maximum number of sampling attempts before raising an exception
        when replacement=False.

    Attributes
    ----------
    visited_positions : set or None
        Stores already visited positions when replacement=False.
    """

    def __init__(
        self, 
        ss_positions: List[List[Any]], 
        replacement: bool = True,
        max_attempts: int = 1000
    ):
        self.ss_positions = ss_positions
        self.replacement = replacement
        self.max_attempts = max_attempts
        self.visited_positions: Optional[Set[Tuple]] = set() if not replacement else None

    def ask(self) -> np.ndarray:
        """
        Returns a new random position based on the replacement setting.
        
        When replacement=False, ensures positions are not repeated.

        Returns
        -------
        np.ndarray
            Randomly selected position.
            
        Raises
        ------
        RuntimeError
            If unable to find a new unique position after max_attempts
            when replacement=False.
        """
        if self.replacement:
            return move_random(self.ss_positions)

        # No-replacement behavior
        for attempt in range(self.max_attempts):
            pos = move_random(self.ss_positions)
            pos_tuple = tuple(pos.tolist())  # Hashable for set storage

            if pos_tuple not in self.visited_positions:
                self.visited_positions.add(pos_tuple)
                return pos
                
            if attempt % 100 == 0 and attempt > 0:
                logger.debug(f"Made {attempt} attempts to find unique position")

        # Search space might be exhausted
        search_space_size = np.prod([len(dim) for dim in self.ss_positions])
        visited_count = len(self.visited_positions)
        
        err_msg = (
            f"RandomSearchOptimizer: Failed to find unique position after {self.max_attempts} attempts. "
            f"Visited {visited_count}/{search_space_size} positions. "
            f"Search space might be exhausted."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

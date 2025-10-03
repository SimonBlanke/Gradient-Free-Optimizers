# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class AskTellOptimizer:
    """
    Low-level ask/tell interface wrapper for GFO optimizers.

    Decouples the optimizer from the objective function by providing:
    - ask(): returns a position dictionary to evaluate
    - tell(score): provides the score for the current position

    Parameters
    ----------
    optimizer_class : class
        The optimizer class to wrap (e.g., HillClimbingOptimizer)
    search_space : dict
        Dictionary with parameter names as keys and numpy arrays as values
    init_positions : list of dict
        List of initial position dictionaries to start optimization
    init_scores : list of float
        List of scores corresponding to init_positions
    random_state : int, optional
        Random seed for reproducibility
    constraints : list, optional
        List of constraint functions
    **optimizer_params : dict
        Additional optimizer-specific parameters (e.g., epsilon, n_neighbours)
    """

    def __init__(
        self,
        optimizer_class,
        search_space,
        init_positions,
        init_scores,
        random_state=None,
        constraints=None,
        **optimizer_params
    ):
        if constraints is None:
            constraints = []

        # Validate inputs
        if not isinstance(init_positions, list):
            init_positions = [init_positions]
        if not isinstance(init_scores, list):
            init_scores = [init_scores]

        if len(init_positions) != len(init_scores):
            raise ValueError(
                f"init_positions and init_scores must have same length. "
                f"Got {len(init_positions)} and {len(init_scores)}"
            )

        self.search_space = search_space
        self.init_positions = init_positions
        self.init_scores = init_scores
        self.constraints = constraints

        # Create the underlying optimizer without initialization
        self.optimizer = optimizer_class(
            search_space=search_space,
            initialize={},  # No automatic initialization
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=0,
            nth_process=None,
            **optimizer_params
        )

        # Initialize with provided positions and scores
        self._initialize_with_positions()

        # Track current iteration state
        self._asked = False
        self._current_position = None

    def _initialize_with_positions(self):
        """Initialize optimizer with provided positions and scores."""
        # Convert parameter dictionaries to position arrays
        for para_dict, score in zip(self.init_positions, self.init_scores):
            # Convert dict to position array (para -> value -> position)
            value = self.optimizer.conv.para2value(para_dict)
            position = self.optimizer.conv.value2position(value)

            # Track the position
            self.optimizer.pos_new = position
            self.optimizer.nth_trial += 1

            # Evaluate with the provided score
            self.optimizer.evaluate(score)

        # Set optimizer to iteration mode
        try:
            self.optimizer.finish_initialization()
        except AttributeError:
            # Some optimizers may not have this method
            self.optimizer.search_state = "iter"

    def ask(self):
        """
        Request next position to evaluate.

        Returns
        -------
        dict
            Parameter dictionary to evaluate
        """
        if self._asked:
            raise RuntimeError(
                "Must call tell() before asking for next position"
            )

        # Get next position from optimizer
        position = self.optimizer.iterate()
        self._current_position = position
        self._asked = True

        # Convert position array to parameter dictionary
        value = self.optimizer.conv.position2value(position)
        para_dict = self.optimizer.conv.value2para(value)

        return para_dict

    def tell(self, score):
        """
        Provide score for the current position.

        Parameters
        ----------
        score : float
            Score/fitness value for the position from ask()
        """
        if not self._asked:
            raise RuntimeError(
                "Must call ask() before providing a score with tell()"
            )

        # Update optimizer with score
        self.optimizer.pos_new = self._current_position
        self.optimizer.nth_trial += 1
        self.optimizer.evaluate(score)

        # Reset state
        self._asked = False
        self._current_position = None

    @property
    def best_position(self):
        """Get best position found so far as parameter dictionary."""
        if self.optimizer.pos_best is None:
            return None
        value = self.optimizer.conv.position2value(self.optimizer.pos_best)
        return self.optimizer.conv.value2para(value)

    @property
    def best_score(self):
        """Get best score found so far."""
        return self.optimizer.score_best

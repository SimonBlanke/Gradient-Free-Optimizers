# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Base optimizer with Template Method Pattern for dimension-type-aware iteration.

This module defines the abstract base class that all optimizers must inherit from.
The Template Method Pattern is used to provide a consistent interface for
handling different dimension types (continuous, categorical, discrete-numerical)
while allowing each optimizer to implement its own iteration logic.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers.

    This class defines the Template Method Pattern for dimension-type-aware
    iteration. Subclasses must implement the batch iteration methods for
    each dimension type they support.

    Template Methods (implement to support dimension type):
        _iterate_continuous_batch: For continuous dimensions
        _iterate_categorical_batch: For categorical dimensions
        _iterate_discrete_batch: For discrete-numerical dimensions

    Attributes
    ----------
        search_space: Dictionary mapping parameter names to their search ranges
        initialize: Initialization strategy
        constraints: List of constraint functions
        random_state: Random seed for reproducibility
    """

    name = "Base Optimizer"
    _name_ = "base_optimizer"
    __name__ = "BaseOptimizer"

    optimizer_type = "base"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        """Initialize the optimizer.

        Args:
            search_space: Dictionary mapping parameter names to search ranges.
                - Tuple (min, max): Continuous dimension
                - List [...]: Categorical dimension
                - np.array: Discrete-numerical dimension
            initialize: Initialization strategy (dict or callable)
            constraints: List of constraint functions
            random_state: Random seed for reproducibility
            rand_rest_p: Probability of random restart
            nth_process: Process index for parallel optimization
        """
        # Call super().__init__() for cooperative multiple inheritance
        # This ensures Search.__init__() is called when combined with Search
        super().__init__()

        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.nth_process = nth_process

        # Note: Position/score state is managed by CoreOptimizer with property setters.
        # We don't set them here to avoid triggering setters before lists are created.

        # List of optimizers (for single optimizer, just [self])
        # Population-based optimizers may override this with their population
        self.optimizers = [self]

    def iterate(self):
        """Generate a new position using dimension-type-aware iteration.

        This is the main template method that orchestrates the iteration
        by routing to the appropriate batch methods based on dimension types.

        The orchestration logic should be implemented in CoreOptimizer.

        Returns
        -------
            New position as numpy array
        """
        raise NotImplementedError("Subclasses must implement iterate()")

    @abstractmethod
    def evaluate(self, score_new):
        """Evaluate the new position and update internal state.

        Args:
            score_new: Score from the objective function

        This method should:
        1. Compare score_new with current/best scores
        2. Update pos_current, pos_best, score_current, score_best
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    @abstractmethod
    def _iterate_continuous_batch(self) -> np.ndarray:
        """Generate new values for all continuous dimensions (vectorized).

        Access instance state:
            - self.pos_current[self._continuous_mask]: Current values
            - self._continuous_bounds: Shape (n_continuous, 2) with [min, max]

        Returns
        -------
        np.ndarray
            New values for all continuous dimensions, shape (n_continuous,)
        """
        ...

    @abstractmethod
    def _iterate_categorical_batch(self) -> np.ndarray:
        """Generate new category indices for all categorical dimensions (vectorized).

        Access instance state:
            - self.pos_current[self._categorical_mask]: Current category indices
            - self._categorical_sizes: Number of categories per dimension

        Returns
        -------
        np.ndarray
            New category indices, shape (n_categorical,)
        """
        ...

    @abstractmethod
    def _iterate_discrete_batch(self) -> np.ndarray:
        """Generate new positions for all discrete dimensions (vectorized).

        Access instance state:
            - self.pos_current[self._discrete_mask]: Current positions
            - self._discrete_bounds: Shape (n_discrete, 2) with [0, max_idx]

        Returns
        -------
        np.ndarray
            New positions, shape (n_discrete,)
        """
        ...

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        """Clip position to valid bounds.

        Should be implemented by CoreOptimizer with dimension-type-awareness.

        Args:
            position: Raw position that may be out of bounds

        Returns
        -------
            Clipped position within valid bounds
        """
        raise NotImplementedError("Subclasses must implement _clip_position()")

    @property
    def best_para(self):
        """Return the best parameters found."""
        raise NotImplementedError("Subclasses must implement best_para property")

    # Note: best_score is NOT a property here.
    # Search class sets self.best_score as a regular attribute.
    # score_best (via property in CoreOptimizer) tracks internally with list appends.
    # These are intentionally separate - best_score is for Search's external interface,
    # score_best is for optimizer's internal tracking.

    # Note: search_data is provided by the Search class mixin in optimizer_search/

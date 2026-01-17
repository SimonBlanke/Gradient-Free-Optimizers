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

        # To be set by subclasses or during setup
        self.pos_current = None
        self.pos_best = None
        self.score_current = None
        self.score_best = None

        # List of optimizers (for single optimizer, just [self])
        # Population-based optimizers may override this with their population
        self.optimizers = [self]

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHOD: ORCHESTRATION
    # Note: search() is provided by the Search class mixin in optimizer_search/
    # ═══════════════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPLATE METHODS: DIMENSION-TYPE-SPECIFIC ITERATION
    # These methods define the "extension points" of the Template Method Pattern.
    # Subclasses implement these to support specific dimension types.
    # ═══════════════════════════════════════════════════════════════════════════

    def _iterate_continuous_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Iterate ALL continuous dimensions at once (vectorized).

        This method must be implemented by subclasses that support
        continuous dimensions.

        Args:
            current: Current values of all continuous dimensions.
                Shape: (n_continuous,)
            bounds: Min/max bounds for each continuous dimension.
                Shape: (n_continuous, 2) where [:, 0] is min and [:, 1] is max

        Returns
        -------
            New values for all continuous dimensions.
            Shape: (n_continuous,)

        Raises
        ------
            NotImplementedError: If continuous dimensions are not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement continuous dimension "
            f"support. Implement _iterate_continuous_batch() to enable this feature."
        )

    def _iterate_categorical_batch(
        self,
        current: np.ndarray,
        n_categories: np.ndarray,
    ) -> np.ndarray:
        """Iterate ALL categorical dimensions at once (vectorized).

        This method must be implemented by subclasses that support
        categorical dimensions.

        Args:
            current: Current category indices for all categorical dimensions.
                Shape: (n_categorical,)
            n_categories: Number of categories for each categorical dimension.
                Shape: (n_categorical,)

        Returns
        -------
            New category indices for all categorical dimensions.
            Shape: (n_categorical,)

        Raises
        ------
            NotImplementedError: If categorical dimensions are not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement categorical dimension "
            f"support. Implement _iterate_categorical_batch() to enable this feature."
        )

    def _iterate_discrete_batch(
        self,
        current: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Iterate ALL discrete-numerical dimensions at once (vectorized).

        This method must be implemented by subclasses that support
        discrete-numerical dimensions.

        Args:
            current: Current positions for all discrete dimensions.
                Shape: (n_discrete,)
            bounds: Min/max bounds (as indices) for each discrete dimension.
                Shape: (n_discrete, 2) where [:, 0] is min and [:, 1] is max

        Returns
        -------
            New positions for all discrete dimensions.
            Shape: (n_discrete,)

        Raises
        ------
            NotImplementedError: If discrete dimensions are not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement discrete-numerical "
            f"dimension support. Implement _iterate_discrete_batch() to enable "
            f"this feature."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════

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

    @property
    def best_score(self):
        """Return the best score found."""
        return self.score_best

    @best_score.setter
    def best_score(self, value):
        """Set the best score."""
        self.score_best = value

    # Note: search_data is provided by the Search class mixin in optimizer_search/

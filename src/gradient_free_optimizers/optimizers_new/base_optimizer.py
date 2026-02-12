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

from .core_optimizer import CoreOptimizer


class BaseOptimizer(CoreOptimizer):
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

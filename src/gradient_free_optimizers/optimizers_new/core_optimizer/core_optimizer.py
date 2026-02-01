# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Core optimizer that implements the iteration orchestration logic.

This class extends BaseOptimizer with the actual orchestration of
dimension-type-aware iteration using masks and batch operations.

State Management:
    Uses property-based state tracking (like SearchTracker). Setting
    pos_new, pos_current, pos_best, score_new, score_current, score_best
    automatically appends to the corresponding history lists.

Template Method Pattern:
    Subclasses implement _iterate_*_batch() methods WITHOUT parameters.
    They access state via self.pos_current, self._continuous_bounds, etc.
"""

import math

import numpy as np

from ..base_optimizer import BaseOptimizer
from .converter import Converter
from .init_positions import Initializer
from .utils import set_random_seed


def _isinf(x):
    """Check if value is infinite."""
    return math.isinf(x) if isinstance(x, int | float) else np.isinf(x)


def _isnan(x):
    """Check if value is NaN."""
    return math.isnan(x) if isinstance(x, int | float) else np.isnan(x)


class CoreOptimizer(BaseOptimizer):
    """Core optimizer with iteration orchestration.

    This class implements the orchestration logic for the Template Method Pattern.
    It handles:
    - Dimension type detection and mask creation
    - Routing to appropriate batch iteration methods
    - Position clipping and validation
    - Converter and Initializer integration for Search compatibility

    Subclasses should implement the _iterate_*_batch() methods.
    """

    name = "Core Optimizer"
    _name_ = "core_optimizer"
    __name__ = "CoreOptimizer"

    def __init__(
        self,
        search_space,
        initialize=None,
        constraints=None,
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        # Set random seed for reproducibility
        self.random_seed = set_random_seed(nth_process, random_state)

        # Initialize Converter and Initializer for Search compatibility
        self.conv = Converter(search_space, constraints if constraints else [])
        self.init = Initializer(self.conv, initialize if initialize else {"random": 1})

        # Dimension masks - to be initialized during setup
        self._continuous_mask = None
        self._categorical_mask = None
        self._discrete_mask = None

        # Bounds arrays for vectorized operations
        self._continuous_bounds = None
        self._categorical_sizes = None
        self._discrete_bounds = None

        # Tracking state for evaluate()
        self.scores_valid = []
        self.positions_valid = []
        self.nth_trial = 0
        self.nth_init = 0

        # History lists (populated by property setters automatically)
        self.pos_new_list = []
        self.score_new_list = []
        self.pos_current_list = []
        self.score_current_list = []
        self.pos_best_list = []
        self.score_best_list = []

        # Private backing fields for property-based state management
        self._pos_new = None
        self._pos_current = None
        self._pos_best = None
        self._score_new = -math.inf
        self._score_current = -math.inf
        self._score_best = -math.inf

        # Search state
        self.search_state = "init"
        self.best_since_iter = 0

        # Auto-initialize dimension masks
        self._setup_dimension_masks()

    def _setup_dimension_masks(self):
        """Initialize dimension masks and bounds arrays.

        This should be called after the converter is initialized.
        Creates boolean masks for each dimension type and extracts
        bounds/sizes as contiguous arrays for vectorized operations.

        The masks allow efficient selection of dimensions by type:
            position[self._continuous_mask] -> all continuous values

        The bounds arrays are shaped for vectorized operations:
            _continuous_bounds: shape (n_continuous, 2) with [min, max]
            _categorical_sizes: shape (n_categorical,) with n_categories
            _discrete_bounds: shape (n_discrete, 2) with [0, max_index]
        """
        n_dims = len(self.search_space)
        dim_names = list(self.search_space.keys())

        # Initialize masks as boolean arrays
        continuous_mask = np.zeros(n_dims, dtype=bool)
        categorical_mask = np.zeros(n_dims, dtype=bool)
        discrete_mask = np.zeros(n_dims, dtype=bool)

        # Lists to collect bounds/sizes for each type
        continuous_bounds_list = []
        categorical_sizes_list = []
        discrete_bounds_list = []

        # Classify each dimension
        for i, name in enumerate(dim_names):
            dim_def = self.search_space[name]

            if isinstance(dim_def, tuple) and len(dim_def) == 2:
                # Continuous: tuple (min, max)
                continuous_mask[i] = True
                continuous_bounds_list.append([dim_def[0], dim_def[1]])

            elif isinstance(dim_def, list):
                # Categorical: list of options
                categorical_mask[i] = True
                categorical_sizes_list.append(len(dim_def))

            elif isinstance(dim_def, np.ndarray):
                # Discrete numerical: numpy array
                discrete_mask[i] = True
                discrete_bounds_list.append([0, len(dim_def) - 1])

            else:
                raise ValueError(
                    f"Unknown dimension type for '{name}': {type(dim_def)}. "
                    f"Expected tuple (continuous), list (categorical), "
                    f"or np.ndarray (discrete)."
                )

        # Store masks
        self._continuous_mask = continuous_mask
        self._categorical_mask = categorical_mask
        self._discrete_mask = discrete_mask

        # Convert bounds/sizes to numpy arrays for vectorized operations
        self._continuous_bounds = (
            np.array(continuous_bounds_list) if continuous_bounds_list else None
        )
        self._categorical_sizes = (
            np.array(categorical_sizes_list) if categorical_sizes_list else None
        )
        self._discrete_bounds = (
            np.array(discrete_bounds_list) if discrete_bounds_list else None
        )

    def init_pos(self):
        """Get next initialization position.

        Returns the next position from the initialization list and tracks it
        as the new position. Called by Search during the initialization phase.

        Do not override this method. Override _init_pos() instead to add
        algorithm-specific initialization logic.

        Note: Property setter automatically appends to pos_new_list.

        Returns
        -------
        np.ndarray
            The next initialization position.
        """
        init_pos = self.init.init_positions_l[self.nth_init]
        self.pos_new = init_pos  # Property setter auto-appends
        self.nth_init += 1

        # Call algorithm-specific hook
        self._init_pos(init_pos)

        return init_pos

    def _init_pos(self, position):
        """Algorithm-specific initialization position logic.

        Override this method to perform algorithm-specific setup when
        a new initialization position is generated. This is called by
        init_pos() after the position is tracked.

        For population-based optimizers, this can be used to:
        - Select which sub-optimizer receives this position
        - Initialize sub-optimizer state
        - Track positions across multiple individuals

        Args:
            position: The initialization position that was just generated

        Default implementation does nothing.
        """
        pass

    def evaluate_init(self, score_new):
        """Handle initialization phase evaluation.

        Updates best and current positions/scores based on the initialization
        evaluation. Called by Search after evaluating an init position.

        Do not override this method. Override _evaluate_init() instead to add
        algorithm-specific evaluation logic during initialization.

        Note: Property setters automatically append to history lists and
        track valid scores.

        Args:
            score_new: Score of the most recently evaluated init position
        """
        # Track the score (property setter auto-appends and tracks valid)
        self.score_new = score_new

        # Initialize best if first evaluation or better score
        if self._pos_best is None or score_new > self._score_best:
            self.pos_best = self._pos_new.copy()
            self.score_best = score_new

        # Initialize current if first evaluation
        if self._pos_current is None:
            self.pos_current = self._pos_new.copy()
            self.score_current = score_new

        self.nth_trial += 1

        # Call algorithm-specific hook
        self._evaluate_init(score_new)

    def _evaluate_init(self, score_new):
        """Algorithm-specific initialization evaluation logic.

        Override this method to perform algorithm-specific processing when
        an initialization position is evaluated. This is called by
        evaluate_init() after standard tracking is done.

        For population-based optimizers, this can be used to:
        - Delegate evaluation to current sub-optimizer
        - Update population-wide best tracking
        - Perform algorithm-specific bookkeeping

        Args:
            score_new: Score of the most recently evaluated init position

        Default implementation does nothing.
        """
        pass

    def finish_initialization(self):
        """Transition from initialization to iteration phase.

        Called by Search after all init positions have been evaluated.
        Sets the search state to "iter" for the iteration phase.

        DO NOT OVERRIDE THIS METHOD. Override _finish_initialization() instead
        to add algorithm-specific initialization logic.
        """
        # Call algorithm-specific hook for initialization finalization
        self._finish_initialization()
        self.search_state = "iter"

    def _finish_initialization(self):
        """Perform algorithm-specific setup after init phase.

        Override this method to perform any algorithm-specific setup that
        needs to happen after all init positions have been evaluated but
        before the iteration phase begins.

        Examples
        --------
        - DownhillSimplex: Initialize simplex vertices from evaluated positions
        - PopulationOptimizers: Sort population by fitness

        Default implementation does nothing.
        """
        pass

    def iterate(self):
        """Generate a new position using dimension-type-aware batch iteration.

        This method orchestrates the iteration by:
        1. Creating an empty position array
        2. Calling batch methods for each dimension type that has elements
        3. Clipping the result to valid bounds
        4. Checking constraints (regenerate if violated)

        Note: Property setter automatically appends to pos_new_list.

        Returns
        -------
            New position as numpy array
        """
        # Retry loop for constraint satisfaction
        max_retries = 100
        for _ in range(max_retries):
            clipped_pos = self._generate_position()

            # Check constraints
            if self.conv.not_in_constraint(clipped_pos):
                break
        # If max retries exceeded, use the last generated position anyway

        # Track as new position (property setter auto-appends)
        self.pos_new = clipped_pos

        return clipped_pos

    def _generate_position(self):
        """Generate a single candidate position (internal helper).

        Calls the template methods (_iterate_*_batch) WITHOUT parameters.
        Each method accesses state via self.pos_current, self._*_bounds, etc.

        Returns
        -------
            Clipped position as numpy array
        """
        n_dims = len(self.search_space)
        new_pos = np.empty(n_dims)

        # Process continuous dimensions
        if self._continuous_mask is not None and self._continuous_mask.any():
            new_pos[self._continuous_mask] = self._iterate_continuous_batch()

        # Process categorical dimensions
        if self._categorical_mask is not None and self._categorical_mask.any():
            new_pos[self._categorical_mask] = self._iterate_categorical_batch()

        # Process discrete-numerical dimensions
        if self._discrete_mask is not None and self._discrete_mask.any():
            new_pos[self._discrete_mask] = self._iterate_discrete_batch()

        return self._clip_position(new_pos)

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        """Clip position to valid bounds with dimension-type-awareness.

        For continuous: clip to (min, max) range
        For categorical: clip to [0, n_categories-1] and cast to int
        For discrete: clip to [0, max_index] and cast to int

        Args:
            position: Raw position that may be out of bounds

        Returns
        -------
            Clipped position within valid bounds
        """
        clipped = position.copy()

        # Clip continuous dimensions to their bounds
        if self._continuous_mask is not None and self._continuous_mask.any():
            cont_vals = clipped[self._continuous_mask]
            mins = self._continuous_bounds[:, 0]
            maxs = self._continuous_bounds[:, 1]
            clipped[self._continuous_mask] = np.clip(cont_vals, mins, maxs)

        # Clip categorical dimensions to valid indices [0, n_categories-1]
        if self._categorical_mask is not None and self._categorical_mask.any():
            cat_vals = clipped[self._categorical_mask]
            # Round to nearest integer and clip to valid range
            cat_vals = np.round(cat_vals).astype(np.int64)
            cat_vals = np.clip(cat_vals, 0, self._categorical_sizes - 1)
            clipped[self._categorical_mask] = cat_vals

        # Clip discrete dimensions to valid indices [0, max_index]
        if self._discrete_mask is not None and self._discrete_mask.any():
            disc_vals = clipped[self._discrete_mask]
            # Round to nearest integer and clip to valid range
            disc_vals = np.round(disc_vals).astype(np.int64)
            mins = self._discrete_bounds[:, 0].astype(np.int64)
            maxs = self._discrete_bounds[:, 1].astype(np.int64)
            disc_vals = np.clip(disc_vals, mins, maxs)
            clipped[self._discrete_mask] = disc_vals

        return clipped

    def evaluate(self, score_new):
        """Orchestrate evaluation: track score, delegate to algorithm-specific logic.

        This method implements the Template Method Pattern for evaluation:
        1. Common tracking (scores, positions, trial count)
        2. Algorithm-specific acceptance/update logic via _evaluate()

        This method handles both initialization and iteration phases for
        backward compatibility with the backend API.

        Args:
            score_new: Score of the most recently evaluated position
        """
        self._track_score(score_new)

        # Handle initialization phase (first evaluation or pos_best is None)
        # Property setters auto-append to lists, so no manual append needed
        if self.pos_best is None:
            self.pos_best = self.pos_new.copy()
            self.score_best = score_new
            self.best_since_iter = self.nth_trial

        if self.pos_current is None:
            self.pos_current = self.pos_new.copy()
            self.score_current = score_new

        # Delegate to algorithm-specific evaluation
        self._evaluate(score_new)

    def _track_score(self, score_new):
        """Track score and position in history (common to all optimizers).

        Note: Property setter automatically appends to history and tracks
        valid (non-inf, non-nan) scores.

        Args:
            score_new: Score of the most recently evaluated position
        """
        # Property setter handles all tracking automatically
        self.score_new = score_new
        self.nth_trial += 1

    def _evaluate(self, score_new):
        """Algorithm-specific evaluation logic. Override in subclass.

        This method should implement the acceptance criteria and state updates
        specific to each optimization algorithm:
        - Hill Climbing: greedy acceptance (accept if better)
        - Simulated Annealing: probabilistic acceptance based on temperature
        - PSO: update personal best
        - DE: replace parent if offspring is better

        Args:
            score_new: Score of the most recently evaluated position
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _evaluate()"
        )

    def _update_current(self, position, score):
        """Update the current position and score.

        Note: Property setters automatically append to history lists.
        """
        self.pos_current = position.copy()
        self.score_current = score

    def _update_best(self, position, score):
        """Update the best position if this score is better.

        Note: Property setters automatically append to history lists.
        """
        if self._score_best == -math.inf or score > self._score_best:
            self.pos_best = position.copy()
            self.score_best = score

    @property
    def pos_new(self):
        """Get the newest position."""
        return self._pos_new

    @pos_new.setter
    def pos_new(self, pos):
        """Set new position and auto-append to history."""
        self.pos_new_list.append(pos)
        self._pos_new = pos

    @property
    def score_new(self):
        """Get the newest score."""
        return self._score_new

    @score_new.setter
    def score_new(self, score):
        """Set new score and auto-append to history.

        Also tracks valid (non-inf, non-nan) scores.
        """
        self.score_new_list.append(score)
        self._score_new = score
        # Auto-track valid scores
        if not (_isinf(score) or _isnan(score)):
            self.positions_valid.append(self._pos_new)
            self.scores_valid.append(score)

    @property
    def pos_current(self):
        """Get the current position."""
        return self._pos_current

    @pos_current.setter
    def pos_current(self, pos):
        """Set current position and auto-append to history."""
        self.pos_current_list.append(pos)
        self._pos_current = pos

    @property
    def score_current(self):
        """Get the current score."""
        return self._score_current

    @score_current.setter
    def score_current(self, score):
        """Set current score and auto-append to history."""
        self.score_current_list.append(score)
        self._score_current = score

    @property
    def pos_best(self):
        """Get the best position."""
        return self._pos_best

    @pos_best.setter
    def pos_best(self, pos):
        """Set best position and auto-append to history."""
        self.pos_best_list.append(pos)
        self._pos_best = pos

    @property
    def score_best(self):
        """Get the best score."""
        return self._score_best

    @score_best.setter
    def score_best(self, score):
        """Set best score and auto-append to history.

        Also updates best_since_iter.
        """
        self.score_best_list.append(score)
        self._score_best = score
        self.best_since_iter = self.nth_trial

    @property
    def best_para(self):
        """Return the best parameters found as a dictionary.

        Uses the Converter to transform the best position into
        user-friendly parameter names and values.

        Returns
        -------
        dict or None
            Dictionary mapping parameter names to their best values,
            or None if no evaluation has been performed yet.
        """
        # If explicitly set, return that value
        if hasattr(self, "_best_para") and self._best_para is not None:
            return self._best_para
        # Otherwise compute from pos_best
        if self.pos_best is None:
            return None
        best_value = self.conv.position2value(self.pos_best)
        return self.conv.value2para(best_value)

    @best_para.setter
    def best_para(self, value):
        """Set the best parameters explicitly."""
        self._best_para = value

    @property
    def best_value(self):
        """Return the best values found (raw parameter values).

        Returns
        -------
        list or None
            List of best values in parameter order,
            or None if no evaluation has been performed yet.
        """
        # If explicitly set, return that value
        if hasattr(self, "_best_value") and self._best_value is not None:
            return self._best_value
        # Otherwise compute from pos_best
        if self.pos_best is None:
            return None
        return self.conv.position2value(self.pos_best)

    @best_value.setter
    def best_value(self, value):
        """Set the best value explicitly."""
        self._best_value = value

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import random
from collections.abc import Callable
from functools import wraps
from typing import Any

from gradient_free_optimizers._array_backend import (
    array,
    clip,
    rint,
)
from gradient_free_optimizers._array_backend import (
    random as np_random,
)
from gradient_free_optimizers._dimension_types import DimensionType

from .converter import ArrayLike, Converter
from .init_positions import Initializer
from .search_tracker import SearchTracker
from .utils import move_random, set_random_seed


def _get_dist_func(name: str) -> Callable:
    """Get distribution function from array backend."""
    dist_map = {
        "normal": np_random.normal,
        "laplace": np_random.laplace,
        "logistic": np_random.logistic,
        "gumbel": np_random.gumbel,
    }
    return dist_map[name]


class CoreOptimizer(SearchTracker):
    """
    Core optimization mechanics for position generation and evaluation.

    CoreOptimizer provides the fundamental building blocks that all
    optimization algorithms use: position tracking, search space conversion,
    initialization handling, and movement utilities. It bridges the gap
    between abstract optimization logic and concrete array manipulations.

    This class manages:

    - **Position Tracking**: Via inheritance from SearchTracker, maintains
      current, new, and best positions with their scores
    - **Search Space Conversion**: The ``conv`` (Converter) object handles
      transformations between positions (array indices), values (actual
      parameter values), and parameters (dict format)
    - **Initialization**: The ``init`` (Initializer) object generates
      starting positions based on the initialization strategy
    - **Movement Utilities**: Methods like ``move_climb`` and ``move_random``
      for generating candidate positions

    The position representation uses integer indices into the search space
    arrays, which enables efficient constraint checking and discrete
    optimization. The Converter handles mapping these to actual values.

    Parameters
    ----------
    search_space : dict[str, array-like]
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict
        Initialization configuration passed to Initializer.
    constraints : list[callable]
        Constraint functions passed to Converter.
    random_state : int or None
        Random seed for reproducibility.
    rand_rest_p : float
        Probability of random restart (used by ``random_iteration`` decorator).
    nth_process : int or None
        Process identifier for parallel scenarios.

    Attributes
    ----------
    conv : Converter
        Handles position/value/parameter conversions and constraint checking.
    init : Initializer
        Generates initial positions based on the initialization strategy.
    nth_init : int
        Counter for initialization steps completed.
    nth_trial : int
        Counter for total evaluations (init + iterations).
    search_state : str
        Either "init" (initialization phase) or "iter" (iteration phase).

    See Also
    --------
    SearchTracker : Tracks positions and scores throughout optimization.
    Converter : Handles search space transformations.
    Initializer : Generates initial positions.
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int],
        constraints: list[Callable[[dict[str, Any]], bool]] | None,
        random_state: int | None,
        rand_rest_p: float,
        nth_process: int | None,
    ) -> None:
        super().__init__()

        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints if constraints is not None else []
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.nth_process = nth_process

        self.random_seed = set_random_seed(self.nth_process, self.random_state)

        self.conv = Converter(self.search_space, self.constraints)
        self.init = Initializer(self.conv, self.initialize)

        self.nth_init = 0
        self.nth_trial = 0
        self.search_state = "init"

    @staticmethod
    def random_iteration(func: Callable) -> Callable:
        """Decorator that randomly replaces iteration with random exploration.

        With probability ``rand_rest_p``, returns a random position instead of
        calling the wrapped iteration method. This helps escape local optima.
        """

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> ArrayLike:
            if self.rand_rest_p > random.uniform(0, 1):
                return self.move_random()
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def move_climb(
        self,
        pos: ArrayLike,
        epsilon: float = 0.03,
        distribution: str = "normal",
        epsilon_mod: float = 1,
    ) -> ArrayLike:
        dist_func = _get_dist_func(distribution)
        while True:
            sigma = self.conv.max_positions * epsilon * epsilon_mod
            pos_normal = dist_func(pos, sigma, pos.shape)
            pos = self.conv2pos(pos_normal)

            if self.conv.not_in_constraint(pos):
                return pos
            epsilon_mod *= 1.01

    def conv2pos(self, pos: ArrayLike) -> ArrayLike:
        """Convert and clip position to valid integer indices.

        Rounds positions to integers and clips to search space bounds.
        If the position was far outside the bounds (average clipping > 50%
        of dimension size), returns a random position to avoid wall-sticking.
        """
        # position to int
        r_pos = rint(pos)

        n_zeros = [0] * len(self.conv.max_positions)
        # clip into search space boundaries
        pos_clipped = clip(r_pos, n_zeros, self.conv.max_positions).astype(int)

        # Check if position was far outside bounds (wall-sticking prevention)
        # Use average relative clip per dimension - works for any n_dimensions
        clip_amounts = [abs(r_pos[i] - pos_clipped[i]) for i in range(len(r_pos))]
        dim_sizes = [max(1, m) for m in self.conv.max_positions]  # Avoid div by 0
        relative_clips = [c / s for c, s in zip(clip_amounts, dim_sizes)]
        avg_relative_clip = sum(relative_clips) / len(relative_clips)

        # If average clip > 50% of dimension size, position was way outside
        if avg_relative_clip > 0.5:
            return self.move_random()

        return pos_clipped

    def move_random(self) -> ArrayLike:
        while True:
            pos = move_random(self.conv.search_space_positions)
            if self.conv.not_in_constraint(pos):
                return pos

    @SearchTracker.track_new_pos
    def init_pos(self) -> ArrayLike:
        init_pos = self.init.init_positions_l[self.nth_init]
        return init_pos

    @SearchTracker.track_new_score
    def evaluate_init(self, score_new: float) -> None:
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.score_best = score_new

        if self.pos_current is None:
            self.pos_current = self.pos_new
            self.score_current = score_new

    # ═══════════════════════════════════════════════════════════════════════
    # TYPE-AWARE METHODS FOR EXTENDED SEARCH SPACE SUPPORT
    # These methods handle discrete, continuous, and categorical dimensions.
    # In legacy mode (all discrete-numerical), they delegate to original methods.
    # ═══════════════════════════════════════════════════════════════════════

    def move_climb_typed(
        self,
        pos: ArrayLike,
        epsilon: float = 0.03,
        distribution: str = "normal",
        epsilon_mod: float = 1,
    ) -> ArrayLike:
        """Type-aware hill climbing movement.

        In legacy mode (all discrete-numerical dimensions), delegates to
        the original move_climb method. For mixed dimension types, handles
        each type appropriately:
        - Discrete numerical: Gaussian noise scaled by dimension size
        - Continuous: Gaussian noise scaled by range
        - Categorical: Probabilistic category switch

        Parameters
        ----------
        pos : array-like
            Current position to move from.
        epsilon : float
            Step size parameter (0 to 1).
        distribution : str
            Noise distribution ("normal", "laplace", "logistic", "gumbel").
        epsilon_mod : float
            Modifier for epsilon, increased on constraint violations.

        Returns
        -------
        array-like
            New position satisfying all constraints.
        """
        if self.conv.is_legacy_mode:
            return self.move_climb(pos, epsilon, distribution, epsilon_mod)

        return self._move_climb_mixed_types(pos, epsilon, distribution, epsilon_mod)

    def _move_climb_mixed_types(
        self,
        pos: ArrayLike,
        epsilon: float,
        distribution: str,
        epsilon_mod: float,
    ) -> ArrayLike:
        """Internal implementation for mixed dimension types."""
        dist_func = _get_dist_func(distribution)

        while True:
            new_pos = []

            for idx, dim_type in enumerate(self.conv.dim_types):
                if dim_type == DimensionType.DISCRETE_NUMERICAL:
                    # Gaussian noise scaled by dimension size
                    max_pos = self.conv.dim_infos[idx].bounds[1]
                    sigma = max_pos * epsilon * epsilon_mod
                    noise = dist_func(0, sigma)
                    new_val = pos[idx] + noise
                    new_pos.append(new_val)

                elif dim_type == DimensionType.CONTINUOUS:
                    # Gaussian noise scaled by range
                    min_val, max_val = self.conv.dim_infos[idx].bounds
                    range_size = max_val - min_val
                    sigma = range_size * epsilon * epsilon_mod
                    noise = dist_func(0, sigma)
                    new_val = pos[idx] + noise
                    new_pos.append(new_val)

                elif dim_type == DimensionType.CATEGORICAL:
                    # Probabilistic category switch
                    if random.random() < epsilon * epsilon_mod:
                        max_idx = self.conv.dim_infos[idx].bounds[1]
                        new_pos.append(random.randint(0, max_idx))
                    else:
                        new_pos.append(pos[idx])

            new_pos = self.conv2pos_typed(array(new_pos))

            if self.conv.not_in_constraint(new_pos):
                return new_pos

            epsilon_mod *= 1.01

    def move_random_typed(self) -> ArrayLike:
        """Type-aware random position generation.

        In legacy mode, delegates to the original move_random method.
        For mixed dimension types, generates appropriate random values
        for each dimension type.

        Returns
        -------
        array-like
            Random position satisfying all constraints.
        """
        if self.conv.is_legacy_mode:
            return self.move_random()

        return self._move_random_mixed_types()

    def _move_random_mixed_types(self) -> ArrayLike:
        """Internal implementation for mixed dimension types."""
        while True:
            pos = []

            for idx, dim_type in enumerate(self.conv.dim_types):
                bounds = self.conv.dim_infos[idx].bounds

                if dim_type == DimensionType.CONTINUOUS:
                    # Uniform random in continuous range
                    pos.append(random.uniform(bounds[0], bounds[1]))
                else:
                    # Random index for discrete/categorical
                    pos.append(random.randint(int(bounds[0]), int(bounds[1])))

            pos = array(pos)

            if self.conv.not_in_constraint(pos):
                return pos

    def conv2pos_typed(self, pos: ArrayLike) -> ArrayLike:
        """Type-aware position conversion and clipping.

        In legacy mode, delegates to the original conv2pos method.
        For mixed dimension types, handles each type appropriately:
        - Discrete numerical: Round and clip to valid indices
        - Continuous: Clip to bounds (no rounding)
        - Categorical: Round and clip to valid category indices

        Parameters
        ----------
        pos : array-like
            Position to convert/clip.

        Returns
        -------
        array-like
            Valid position within search space bounds.
        """
        if self.conv.is_legacy_mode:
            return self.conv2pos(pos)

        return self._conv2pos_mixed_types(pos)

    def _conv2pos_mixed_types(self, pos: ArrayLike) -> ArrayLike:
        """Internal implementation for mixed dimension types."""
        clipped = []

        for idx, dim_type in enumerate(self.conv.dim_types):
            bounds = self.conv.dim_infos[idx].bounds
            val = pos[idx]

            if dim_type == DimensionType.CONTINUOUS:
                # Clip to bounds, keep as float
                clipped_val = max(bounds[0], min(bounds[1], float(val)))
                clipped.append(clipped_val)
            else:
                # Round and clip to integer indices
                rounded = round(val)
                clipped_val = int(max(bounds[0], min(bounds[1], rounded)))
                clipped.append(clipped_val)

        return array(clipped)

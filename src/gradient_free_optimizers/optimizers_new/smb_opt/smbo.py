# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Sequential Model-Based Optimization (SMBO) base class.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from ..core_optimizer import CoreOptimizer
from .sampling import InitialSampler

if TYPE_CHECKING:
    import pandas as pd


class SMBO(CoreOptimizer):
    """Base class for Sequential Model-Based Optimization.

    Dimension Support:
        - Continuous: YES (surrogate model based)
        - Categorical: YES (with index encoding)
        - Discrete: YES (surrogate model based)

    SMBO algorithms build a surrogate model of the objective function
    and use an acquisition function to select the next point to evaluate.
    The surrogate model is trained on evaluated positions and their scores.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
        Strategy for generating initial positions before model-based search.
    constraints : list, optional
        List of constraint functions that filter valid positions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of performing a random iteration instead of model-based.
    nth_process : int, optional
        Process index for parallel optimization.
    warm_start_smbo : pd.DataFrame, optional
        Previous optimization results to initialize the surrogate model.
    max_sample_size : int, default=10000000
        Maximum number of positions to consider for sampling.
    sampling : dict, False, or None, default=None
        Sampling strategy for large search spaces. Use False to disable.
    replacement : bool, default=True
        Whether to allow re-evaluation of the same position.

    Attributes
    ----------
    X_sample : list
        List of evaluated positions (training data for surrogate).
    Y_sample : list
        List of scores corresponding to X_sample.
    """

    name = "SMBO"
    _name_ = "smbo"
    __name__ = "SMBO"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        warm_start_smbo: pd.DataFrame | None = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] | Literal[False] | None = None,
        replacement: bool = True,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling if sampling is not None else {"random": 1000000}
        self.replacement = replacement

        # Create sampler for candidate position generation
        self.sampler = InitialSampler(self.conv, max_sample_size)

        # Initialize training data
        self.init_warm_start_smbo(warm_start_smbo)

        # Will be populated in finish_initialization
        self.all_pos_comb = None

    def init_warm_start_smbo(self, search_data: pd.DataFrame | None) -> None:
        """Initialize X_sample and Y_sample from previous optimization data.

        Filters out invalid values (NaN, inf) and values outside the search space.

        Parameters
        ----------
        search_data : pd.DataFrame or None
            DataFrame with parameter columns and a 'score' column.
        """
        if search_data is not None:
            # Filter out nan and inf
            warm_start_smbo = search_data[
                ~search_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)
            ]

            # Filter out elements that are not in search space
            int_idx_list = []
            for para_name in self.conv.para_names:
                search_data_dim = warm_start_smbo[para_name].values
                search_space_dim = self.conv.search_space[para_name]

                int_idx = np.nonzero(np.isin(search_data_dim, search_space_dim))[0]
                int_idx_list.append(int_idx)

            intersec = int_idx_list[0]
            for int_idx in int_idx_list[1:]:
                intersec = np.intersect1d(intersec, int_idx)
            warm_start_smbo_f = warm_start_smbo.iloc[intersec]

            X_sample_values = warm_start_smbo_f[self.conv.para_names].values
            Y_sample = warm_start_smbo_f["score"].values

            self.X_sample = self.conv.values2positions(X_sample_values)
            self.Y_sample = list(Y_sample)
        else:
            self.X_sample = []
            self.Y_sample = []

    # ═══════════════════════════════════════════════════════════════════════════
    # DECORATORS FOR TRACKING SAMPLES
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def track_X_sample(func: Callable) -> Callable:
        """Decorator that appends returned position to X_sample."""
        def wrapper(self, *args, **kwargs):
            pos = func(self, *args, **kwargs)
            self.X_sample.append(pos)
            return pos
        return wrapper

    @staticmethod
    def track_y_sample(func: Callable) -> Callable:
        """Decorator that appends score to Y_sample, skipping invalid scores."""
        def wrapper(self, score: float) -> None:
            func(self, score)

            if math.isnan(score) if isinstance(score, float) else np.isnan(score):
                del self.X_sample[-1]
            elif math.isinf(score) if isinstance(score, float) else np.isinf(score):
                del self.X_sample[-1]
            else:
                self.Y_sample.append(score)

        return wrapper

    # ═══════════════════════════════════════════════════════════════════════════
    # CANDIDATE POSITION GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _all_possible_pos(self) -> np.ndarray:
        """Generate all possible positions in the search space.

        For large spaces, this is subsampled by _sampling().
        """
        pos_space = self.sampler.get_pos_space()
        n_dim = len(pos_space)

        # Create meshgrid and reshape
        grids = np.meshgrid(*pos_space)
        all_pos_comb = np.array(grids).T.reshape(-1, n_dim)

        # Filter by constraints
        all_pos_comb_constr = []
        for pos in all_pos_comb:
            if self.conv.not_in_constraint(pos):
                all_pos_comb_constr.append(pos)

        return np.array(all_pos_comb_constr)

    def _sampling(self, all_pos_comb: np.ndarray) -> np.ndarray:
        """Subsample candidate positions for large search spaces."""
        if self.sampling is False:
            return all_pos_comb
        elif "random" in self.sampling:
            return self._random_sampling(all_pos_comb)
        return all_pos_comb

    def _random_sampling(self, pos_comb: np.ndarray) -> np.ndarray:
        """Random subsampling of candidate positions."""
        n_samples = self.sampling["random"]
        n_pos_comb = len(pos_comb)

        if n_pos_comb <= n_samples:
            return pos_comb
        else:
            idx_sample = np.random.choice(n_pos_comb, n_samples, replace=False)
            return pos_comb[idx_sample]

    def _remove_position(self, position: np.ndarray) -> None:
        """Remove a position from the candidate set (for replacement=False)."""
        mask = np.all(self.all_pos_comb == position, axis=1)
        self.all_pos_comb = self.all_pos_comb[np.invert(mask)]

    # ═══════════════════════════════════════════════════════════════════════════
    # SEARCH INTERFACE
    # ═══════════════════════════════════════════════════════════════════════════

    def init_pos(self) -> np.ndarray:
        """Get next initialization position and track in X_sample."""
        pos = super().init_pos()
        self.X_sample.append(pos)
        return pos

    def finish_initialization(self) -> None:
        """Transition to iteration phase and generate candidate positions."""
        self.all_pos_comb = self._all_possible_pos()
        super().finish_initialization()

    def iterate(self) -> np.ndarray:
        """Generate next position using surrogate model and acquisition function."""
        pos = self._propose_location()

        self.pos_new = pos
        self.pos_new_list.append(pos)
        self.X_sample.append(pos)

        return pos

    def _propose_location(self) -> np.ndarray:
        """Propose next location using surrogate model.

        Override in subclasses to implement specific SMBO algorithms.
        """
        try:
            self._training()
        except ValueError:
            logging.warning(
                "Training sequential model failed. Performing random iteration."
            )
            return self._move_random()

        exp_imp = self._expected_improvement()

        index_best = list(exp_imp.argsort()[::-1])
        all_pos_comb_sorted = self.pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    def _training(self) -> None:
        """Train the surrogate model on X_sample and Y_sample.

        Override in subclasses.
        """
        raise NotImplementedError("Subclass must implement _training()")

    def _expected_improvement(self) -> np.ndarray:
        """Compute acquisition function values for candidate positions.

        Override in subclasses.
        """
        raise NotImplementedError("Subclass must implement _expected_improvement()")

    def _move_random(self) -> np.ndarray:
        """Generate a random valid position."""
        return self.init.move_random_typed()

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════

    def evaluate(self, score_new: float) -> None:
        """Evaluate the current position and update surrogate training data."""
        self._track_score(score_new)

        # Handle initialization phase
        if self.pos_best is None:
            self.pos_best = self.pos_new.copy()
            self.score_best = score_new
            self.pos_best_list.append(self.pos_best)
            self.score_best_list.append(self.score_best)
            self.best_since_iter = self.nth_trial

        if self.pos_current is None:
            self.pos_current = self.pos_new.copy()
            self.score_current = score_new
            self.pos_current_list.append(self.pos_current)
            self.score_current_list.append(self.score_current)

        # Track Y_sample (skip invalid scores)
        if not (math.isnan(score_new) or math.isinf(score_new)):
            self.Y_sample.append(score_new)
        else:
            # Remove the X_sample entry for invalid scores
            if len(self.X_sample) > len(self.Y_sample):
                del self.X_sample[-1]

        # Update best and current
        self._evaluate(score_new)

        # Remove position if replacement=False
        if not self.replacement and self.pos_new is not None:
            self._remove_position(self.pos_new)

    def evaluate_init(self, score_new: float) -> None:
        """Handle initialization phase evaluation."""
        self._track_score(score_new)

        # Track Y_sample (skip invalid scores)
        if not (math.isnan(score_new) or math.isinf(score_new)):
            self.Y_sample.append(score_new)
        else:
            # Remove the X_sample entry for invalid scores
            if len(self.X_sample) > len(self.Y_sample):
                del self.X_sample[-1]

        # Update best and current
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

        self.nth_trial += 1

    def _evaluate(self, score_new: float) -> None:
        """SMBO evaluation - always update to new position (greedy)."""
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

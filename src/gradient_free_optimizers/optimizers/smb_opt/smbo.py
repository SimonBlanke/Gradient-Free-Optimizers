# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Sequential Model-Based Optimization (SMBO) base class."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..base_optimizer import BaseOptimizer
from ..core_optimizer import CoreOptimizer
from .sampling import InitialSampler

if TYPE_CHECKING:
    import pandas as pd


def _isinf(x):
    """Check if value is infinite."""
    return math.isinf(x) if isinstance(x, int | float) else np.isinf(x)


def _isnan(x):
    """Check if value is NaN."""
    return math.isnan(x) if isinstance(x, int | float) else np.isnan(x)


class SMBO(BaseOptimizer):
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
        self._init_warm_start_smbo(warm_start_smbo)

        # Will be populated in _on_finish_initialization
        self.all_pos_comb = None

        # Cache for surrogate-proposed position (cleared after each iteration)
        self._cached_proposed_pos = None

    def _init_warm_start_smbo(self, search_data: pd.DataFrame | None) -> None:
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

    @property
    def _pos_new(self):
        """Get the newest position."""
        return CoreOptimizer._pos_new.fget(self)

    @_pos_new.setter
    def _pos_new(self, pos):
        """Set new position with SMBO-specific tracking.

        - Clears the cached proposed position (for next iteration)
        - Appends to X_sample (for surrogate training)
        - Delegates standard tracking to parent
        """
        # Clear cache for next iteration
        self._cached_proposed_pos = None
        # Track in X_sample for surrogate training
        self.X_sample.append(pos)
        # Delegate standard tracking to parent
        CoreOptimizer._pos_new.fset(self, pos)

    @property
    def _score_new(self):
        """Get the newest score."""
        return CoreOptimizer._score_new.fget(self)

    @_score_new.setter
    def _score_new(self, score):
        """Set new score with SMBO-specific tracking.

        - Delegates standard tracking to parent
        - Appends to Y_sample for valid scores
        - Removes corresponding X_sample entry for invalid scores
        """
        # Delegate standard tracking to parent
        CoreOptimizer._score_new.fset(self, score)
        # SMBO-specific: track in Y_sample for surrogate training
        if not (_isinf(score) or _isnan(score)):
            self.Y_sample.append(score)
        else:
            # Remove X_sample entry for invalid scores
            if len(self.X_sample) > len(self.Y_sample):
                del self.X_sample[-1]

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

    # NOTE: _init_pos() is NOT overridden
    # X_sample tracking happens via _pos_new setter

    def _on_finish_initialization(self) -> None:
        """Generate candidate positions for surrogate model.

        Called by CoreOptimizer._finish_initialization() after all init
        positions have been evaluated. Generates the candidate position
        grid for the surrogate model to evaluate.

        Note: DO NOT set search_state here - CoreOptimizer handles that.
        """
        self.all_pos_comb = self._all_possible_pos()

    # NOTE: _iterate() is NOT overridden - uses CoreOptimizer._iterate() which calls
    # the _iterate_*_batch() methods below

    def _ensure_surrogate_trained(self) -> None:
        """Train surrogate and cache proposed position if not already done.

        This method is called by each _iterate_*_batch() method. It trains
        the surrogate model once per iteration and caches the proposed position.
        Subsequent calls within the same iteration return immediately.

        The cache is cleared by the _pos_new setter when the position is finalized.
        """
        if self._cached_proposed_pos is not None:
            return

        try:
            self._training()
            exp_imp = self._expected_improvement()
            index_best = list(exp_imp.argsort()[::-1])
            self._cached_proposed_pos = self.pos_comb[index_best[0]]
        except ValueError:
            logging.warning(
                "Training sequential model failed. Performing random iteration."
            )
            self._cached_proposed_pos = self._move_random()

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Return continuous portion of surrogate-proposed position.

        SMBO trains a surrogate model and proposes positions globally,
        then returns the appropriate slice for continuous dimensions.
        """
        self._ensure_surrogate_trained()
        return self._cached_proposed_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Return categorical portion of surrogate-proposed position.

        SMBO trains a surrogate model and proposes positions globally,
        then returns the appropriate slice for categorical dimensions.
        """
        self._ensure_surrogate_trained()
        return self._cached_proposed_pos[self._categorical_mask]

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Return discrete portion of surrogate-proposed position.

        SMBO trains a surrogate model and proposes positions globally,
        then returns the appropriate slice for discrete dimensions.
        """
        self._ensure_surrogate_trained()
        return self._cached_proposed_pos[self._discrete_mask]

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
        """Generate a random valid position.

        When replacement=False, selects from remaining positions in all_pos_comb.
        Raises ValueError if search space is exhausted.
        """
        if not self.replacement:
            if self.all_pos_comb is None or len(self.all_pos_comb) == 0:
                raise ValueError(
                    "Search space exhausted: no positions remaining with "
                    "replacement=False"
                )
            idx = np.random.randint(len(self.all_pos_comb))
            return self.all_pos_comb[idx]
        return self.init.move_random_typed()

    # NOTE: _evaluate() is NOT overridden - uses CoreOptimizer._evaluate()
    # Y_sample tracking happens via _score_new setter

    # NOTE: _evaluate_init() is NOT overridden - uses CoreOptimizer._evaluate_init()
    # Y_sample tracking happens via _score_new setter

    def _on_evaluate(self, score_new: float) -> None:
        """SMBO-specific evaluation: update positions and handle replacement.

        This template method is called by CoreOptimizer._evaluate() after
        common tracking. SMBO uses greedy updates (always moves to new position).

        Note: Y_sample tracking is handled by the _score_new property setter,
        not here, to ensure it works for both init and iterate phases.

        Args:
            score_new: Score of the most recently evaluated position
        """
        # Update best and current (greedy - always accept new position)
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        # Remove position from candidates if replacement=False
        if not self.replacement and self._pos_new is not None:
            self._remove_position(self._pos_new)

    def _select_diverse_batch(self, exp_imp, n):
        """Select n diverse positions from the acquisition landscape.

        For n=1, returns the single best acquisition position. For n>1,
        clusters the top candidates via MiniBatchKMeans and picks the
        highest-acquisition position from each cluster. This prevents
        batch positions from clumping around a single acquisition peak,
        which would waste parallel evaluations on near-duplicate positions.
        """
        n_candidates = len(exp_imp)

        if n <= 1 or n_candidates <= n:
            top_indices = list(exp_imp.argsort()[::-1][:n])
            return [self.pos_comb[i] for i in top_indices]

        k = min(max(10 * n, 50), n_candidates)
        top_k_indices = exp_imp.argsort()[::-1][:k]
        top_k_positions = np.array([self.pos_comb[i] for i in top_k_indices])
        top_k_ei = exp_imp[top_k_indices]

        if k < 2 * n:
            return [self.pos_comb[i] for i in top_k_indices[:n]]

        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(n, k)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=0)
        labels = kmeans.fit_predict(top_k_positions)

        selected = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_ei = top_k_ei[mask]
            cluster_pos = top_k_positions[mask]
            best_in_cluster = cluster_ei.argmax()
            selected.append(cluster_pos[best_in_cluster])

        return selected

    def _iterate_batch(self, n):
        """Train surrogate once and select n diverse positions."""
        try:
            self._training()
            exp_imp = self._expected_improvement()
            positions = self._select_diverse_batch(exp_imp, n)
            while len(positions) < n:
                positions.append(self._move_random())
            return [self._clip_position(pos) for pos in positions]
        except (ValueError, np.linalg.LinAlgError):
            return [self._clip_position(self._move_random()) for _ in range(n)]

    def _evaluate_batch(self, positions, scores):
        """Process batch results through the standard evaluate chain."""
        for pos, score in zip(positions, scores):
            self._pos_new = pos
            self._evaluate(score)

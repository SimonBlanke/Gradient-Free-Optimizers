# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
DIRECT (DIviding RECTangles) Algorithm.

Supports: CONTINUOUS (auto-discretized), CATEGORICAL, DISCRETE_NUMERICAL
Uses Gower-like distance for mixed dimension types.

Continuous dimensions are automatically discretized using the `resolution`
parameter. This enables systematic rectangle subdivision while maintaining
the algorithm's Lipschitz-based selection properties.

Note: DIRECT is NOT a surrogate-model-based optimizer. It uses systematic
subspace division with Lipschitz bounds, not model training. Therefore it
inherits directly from CoreOptimizer, not from SMBO.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from gradient_free_optimizers._dimension_types import DimensionType
from gradient_free_optimizers._math_backend import cdist

from ..base_optimizer import BaseOptimizer


def _discretize_search_space(search_space, resolution):
    """Convert continuous dimensions to discrete grids for DIRECT.

    Parameters
    ----------
    search_space : dict
        Original search space with potential continuous dimensions (tuples).
    resolution : int
        Number of grid points for continuous dimensions.

    Returns
    -------
    dict
        Search space with continuous dimensions converted to numpy arrays.
    """
    discretized = {}
    for name, space in search_space.items():
        if isinstance(space, tuple) and len(space) == 2:
            # Continuous dimension: convert to linspace
            low, high = space
            if isinstance(low, int | float) and isinstance(high, int | float):
                discretized[name] = np.linspace(low, high, resolution)
            else:
                discretized[name] = space
        else:
            discretized[name] = space
    return discretized


if TYPE_CHECKING:
    import pandas as pd


def _mixed_distance(pos1, pos2, dim_types, dim_infos):
    """Compute Gower-like distance for mixed dimension types.

    For continuous dimensions: normalized Euclidean (diff / range).
    For discrete-numerical dimensions: normalized by dimension size.
    For categorical dimensions: Hamming (0 if same, 1 if different).

    All components are averaged and returned as sqrt of mean squared distance.
    """
    if dim_types is None:
        # Legacy mode: use simple Euclidean
        return cdist(pos1.reshape(1, -1), pos2.reshape(1, -1))[0, 0]

    total_dist = 0.0
    n_dims = len(dim_types)

    for idx, dim_type in enumerate(dim_types):
        if dim_type == DimensionType.CONTINUOUS:
            # Normalized Euclidean for continuous
            range_size = dim_infos[idx].bounds[1] - dim_infos[idx].bounds[0]
            if range_size > 0:
                diff = (pos1[idx] - pos2[idx]) / range_size
            else:
                diff = 0
            total_dist += diff**2
        elif dim_type == DimensionType.CATEGORICAL:
            # Hamming: 0 if same, 1 if different
            total_dist += 0.0 if pos1[idx] == pos2[idx] else 1.0
        else:  # DISCRETE_NUMERICAL
            # Normalized by dimension size
            max_pos = dim_infos[idx].bounds[1]
            if max_pos > 0:
                diff = (pos1[idx] - pos2[idx]) / max_pos
            else:
                diff = 0
            total_dist += diff**2

    return np.sqrt(total_dist / n_dims) if n_dims > 0 else 0.0


class SubSpace:
    """Represents a hyperrectangle in the search space for DIRECT algorithm.

    Each SubSpace tracks its center position, Lipschitz bound, and largest
    dimension for subdivision decisions.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of values for this subspace.
    dim_types : list, optional
        List of DimensionType for each dimension.
    dim_infos : list, optional
        List of DimensionInfo for each dimension.
    """

    def __init__(self, search_space, dim_types=None, dim_infos=None):
        self.search_space = search_space
        self.dim_types = dim_types
        self.dim_infos = dim_infos

        self.score = None
        self.lipschitz_bound = -math.inf
        self.center_pos = self._compute_center_pos()
        self.biggest_dim = self._find_biggest_dim()

    def _compute_center_pos(self):
        """Compute center position of the subspace."""
        center_pos = []

        for dim in list(self.search_space.keys()):
            dim_array = self.search_space[dim]
            array_size = dim_array.shape[0]
            center_idx = int(array_size / 2)
            center_pos.append(dim_array[center_idx])

        return np.array(center_pos)

    def _find_biggest_dim(self):
        """Find the dimension with the largest array (for subdivision)."""
        largest_dim = None
        largest_size = 0

        for dim in list(self.search_space.keys()):
            dim_array = self.search_space[dim]
            array_size = dim_array.shape[0]

            if array_size == largest_size:
                # Random tie-breaking using numpy (respects random_state seed)
                if np.random.randint(0, 2):
                    largest_size = array_size
                    largest_dim = dim
            elif array_size > largest_size:
                largest_size = array_size
                largest_dim = dim

        return largest_dim

    def compute_lipschitz_bound(self, score, K=1):
        """Compute Lipschitz bound using type-aware distance.

        Parameters
        ----------
        score : float
            Function value at center position.
        K : float
            Lipschitz constant estimate.
        """
        self.score = score

        # Furthest corner position
        furthest_pos_ = []
        for dim in list(self.search_space.keys()):
            dim_array = self.search_space[dim]
            furthest_pos_.append(dim_array[0])
        furthest_pos = np.array(furthest_pos_)

        dist = _mixed_distance(
            furthest_pos, self.center_pos, self.dim_types, self.dim_infos
        )

        self.lipschitz_bound = score + K * dist


class DirectAlgorithm(BaseOptimizer):
    """DIRECT algorithm for global optimization.

    Dimension Support:
        - Continuous: YES (rectangle subdivision)
        - Categorical: YES (with Hamming distance)
        - Discrete: YES (normalized distance)

    DIRECT (DIviding RECTangles) systematically divides the search space
    into smaller hyperrectangles and samples their centers. It uses a
    Lipschitz-based selection criterion to identify "potentially optimal"
    rectangles that balance local refinement with global exploration.

    Note: DIRECT is NOT a surrogate-model-based optimizer. Unlike Bayesian
    Optimization or Tree-Parzen Estimators, DIRECT does not train a model.
    Instead, it uses deterministic subspace division with Lipschitz bounds.

    The algorithm proceeds by:
    1. Initialize with the entire search space as one rectangle
    2. Evaluate the center of unevaluated rectangles
    3. Select the rectangle with highest Lipschitz bound
    4. Subdivide it along its largest dimension
    5. Repeat until convergence

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
        Continuous dimensions (tuples) are automatically discretized.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    warm_start : pd.DataFrame, optional
        Previous optimization results to warm-start the algorithm.
        Evaluated positions will have their subspaces marked as scored.
    resolution : int, default=100
        Number of grid points for continuous dimensions. Higher values
        give finer subdivision granularity but increase computation.
    """

    name = "Direct Algorithm"
    _name_ = "direct_algorithm"
    __name__ = "DirectAlgorithm"

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
        warm_start: pd.DataFrame | None = None,
        resolution: int = 100,
        # Legacy SMBO parameters - no-op, kept for backwards compatibility
        warm_start_smbo: pd.DataFrame | None = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] | None = None,
        replacement: bool = True,
    ) -> None:
        # Auto-discretize continuous dimensions before parent init
        self.resolution = resolution
        discretized_space = _discretize_search_space(search_space, resolution)

        super().__init__(
            search_space=discretized_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.warm_start = warm_start
        self.subspace_l: list[SubSpace] = []
        self.current_subspace: SubSpace | None = None

    def _finish_initialization(self) -> None:
        """Initialize with the entire search space as one subspace."""
        subspace = SubSpace(
            self.conv.pos_space,
            dim_types=self.conv.dim_types,
            dim_infos=self.conv.dim_infos,
        )
        self.subspace_l.append(subspace)

    def _select_unevaluated_subspace(self) -> SubSpace | None:
        """Find a subspace that hasn't been evaluated yet."""
        for subspace in self.subspace_l:
            if subspace.score is None:
                return subspace
        return None

    def _select_best_subspace(self) -> SubSpace:
        """Select the subspace with highest Lipschitz bound."""
        lipschitz_bound_max = -math.inf
        next_subspace = None

        for subspace in self.subspace_l:
            if subspace.lipschitz_bound > lipschitz_bound_max:
                lipschitz_bound_max = subspace.lipschitz_bound
                next_subspace = subspace

        # Fallback if all bounds are nan or -inf
        if next_subspace is None:
            next_subspace = self.subspace_l[0]

        return next_subspace

    def _split_subspace(self, subspace: SubSpace, n_splits: int = 3) -> int:
        """Split a subspace along its biggest dimension.

        Parameters
        ----------
        subspace : SubSpace
            The subspace to split.
        n_splits : int
            Number of sub-regions to create.

        Returns
        -------
        int
            Number of new subspaces successfully created.
        """
        search_space = subspace.search_space
        dim_array = search_space[subspace.biggest_dim]

        sub_arrays = np.array_split(dim_array, n_splits)

        sub_search_space_l = []
        for sub_array in sub_arrays:
            sub_search_space_ = dict(search_space)
            sub_search_space_[subspace.biggest_dim] = sub_array
            sub_search_space_l.append(sub_search_space_)

        n_created = 0
        for search_space_ in sub_search_space_l:
            try:
                self.subspace_l.append(
                    SubSpace(
                        search_space_,
                        dim_types=self.conv.dim_types,
                        dim_infos=self.conv.dim_infos,
                    )
                )
                n_created += 1
            except IndexError:
                # IndexError can occur when a dimension has been split into
                # sub-arrays that are too small (e.g., empty or single-element).
                continue

        # Only remove the original subspace if we successfully created new ones
        if n_created > 0:
            self.subspace_l.remove(subspace)

        return n_created

    def _move_random(self) -> np.ndarray:
        """Generate a random valid position using the initializer."""
        return self.init.move_random_typed()

    def _generate_position(self) -> np.ndarray:
        """Generate next position via subspace division.

        DIRECT works by:
        1. Finding an unevaluated subspace and returning its center
        2. If all subspaces are evaluated, split the best one
        3. If no valid subspace found, fall back to random

        Note: Constraint checking is handled by CoreOptimizer.iterate()
        which calls this method in a retry loop.

        Returns
        -------
        np.ndarray
            Next candidate position (may violate constraints).
        """
        # Try to find an unevaluated subspace
        self.current_subspace = self._select_unevaluated_subspace()

        if self.current_subspace is not None:
            return self.current_subspace.center_pos

        # All subspaces evaluated - split the best one
        best_subspace = self._select_best_subspace()
        n_created = self._split_subspace(best_subspace)

        if n_created > 0:
            self.current_subspace = self.subspace_l[-1]
            return self.current_subspace.center_pos

        # Fallback to random position
        self.current_subspace = None
        return self._move_random()

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Not used - DIRECT overrides _generate_position() with subspace division.

        DIRECT does not use the standard batch iteration pattern because
        its algorithm (systematic subspace division) is fundamentally different
        from neighborhood-based search.

        Raises
        ------
        NotImplementedError
            Always raised - see _generate_position() instead.
        """
        raise NotImplementedError(
            "DIRECT uses subspace division, not batch iteration. "
            "See _generate_position() for the algorithm implementation."
        )

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Not used - DIRECT overrides _generate_position()."""
        raise NotImplementedError(
            "DIRECT uses subspace division, not batch iteration. "
            "See _generate_position() for the algorithm implementation."
        )

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Not used - DIRECT overrides _generate_position()."""
        raise NotImplementedError(
            "DIRECT uses subspace division, not batch iteration. "
            "See _generate_position() for the algorithm implementation."
        )

    def _evaluate(self, score_new: float) -> None:
        """Update state after evaluating a position.

        For subspace centers, this updates the Lipschitz bound which is
        used to select the next subspace to subdivide.

        For random positions (when constraints forced a fallback),
        the score is still tracked but no subspace is updated.

        Parameters
        ----------
        score_new : float
            Score for the evaluated position.
        """
        # Update Lipschitz bound for the current subspace (if any)
        if self.current_subspace is not None:
            self.current_subspace.compute_lipschitz_bound(score_new)

        # Update best and current position tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

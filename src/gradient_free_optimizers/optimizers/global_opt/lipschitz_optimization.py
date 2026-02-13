# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Lipschitz Optimization.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gradient_free_optimizers._math_backend import cdist

from ..smb_opt import SMBO

if TYPE_CHECKING:
    import pandas as pd


class LipschitzFunction:
    """Computes Lipschitz-based upper bounds for objective function values.

    Given observed samples (X, Y), estimates the Lipschitz constant from
    pairwise slopes and computes upper bounds at candidate positions.

    Parameters
    ----------
    position_l : array-like
        List of candidate positions to compute bounds for.
    """

    def __init__(self, position_l):
        self.position_l = position_l

    def find_best_slope(self, X_sample, Y_sample):
        """Estimate Lipschitz constant from observed samples.

        Computes the maximum absolute slope between all pairs of samples.

        Parameters
        ----------
        X_sample : list
            List of evaluated positions.
        Y_sample : list
            List of scores at those positions.

        Returns
        -------
        float
            Estimated Lipschitz constant (maximum slope).
        """
        slopes = []

        len_sample = len(X_sample)
        for i in range(len_sample):
            for j in range(i + 1, len_sample):
                x_sample1, y_sample1 = X_sample[i], Y_sample[i]
                x_sample2, y_sample2 = X_sample[j], Y_sample[j]

                if y_sample1 != y_sample2 and np.prod(x_sample1 - x_sample2) != 0:
                    slopes.append(
                        abs(y_sample1 - y_sample2) / abs(x_sample1 - x_sample2)
                    )

        if not slopes:
            return 1
        return np.max(slopes)

    def calculate(self, X_sample, Y_sample, score_best):
        """Compute upper bounds for all candidate positions.

        Uses the estimated Lipschitz constant to bound the possible
        function value at each candidate position based on distance
        to observed samples.

        Parameters
        ----------
        X_sample : list
            List of evaluated positions.
        Y_sample : list
            List of scores at those positions.
        score_best : float
            Best score observed so far.

        Returns
        -------
        np.ndarray
            Upper bounds for each candidate position.
        """
        lip_c = self.find_best_slope(X_sample, Y_sample)

        positions_np = np.array(self.position_l)
        samples_np = np.array(X_sample)

        # Compute distances and scale by Lipschitz constant
        pos_dist = cdist(positions_np, samples_np) * lip_c

        # Upper bound = distance * L + observed value
        upper_bound_l = pos_dist
        upper_bound_l += np.array(Y_sample)

        # Mask zeros and take minimum across samples for each position
        mx = np.ma.masked_array(upper_bound_l, mask=upper_bound_l == 0)
        upper_bound_l = mx.min(1).reshape(1, -1).T

        # Positions that can't improve on best are marked -inf
        upper_bound_l[upper_bound_l <= score_best] = -np.inf

        return upper_bound_l


class LipschitzOptimizer(SMBO):
    """Lipschitz-based global optimizer.

    Dimension Support:
        - Continuous: YES (Lipschitz bounds)
        - Categorical: YES (with appropriate distance metric)
        - Discrete: YES (Lipschitz bounds)

    This optimizer exploits Lipschitz continuity to bound the objective
    function and guide the search. It estimates the Lipschitz constant
    from observed data and uses it to compute upper bounds on the
    objective function across the search space. Points with the highest
    potential (according to these bounds) are selected for evaluation.

    The algorithm:
    1. Estimate Lipschitz constant L from pairwise sample slopes
    2. For each candidate position, compute upper bound using L
    3. Select position with highest upper bound (most potential)
    4. Repeat until convergence

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
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
    warm_start_smbo : pd.DataFrame, optional
        Previous optimization results to initialize the surrogate model.
    max_sample_size : int, default=10000000
        Maximum number of positions to consider for sampling.
    sampling : dict, False, or None, default=None
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Whether to allow re-evaluation of the same position.
    """

    name = "Lipschitz Optimizer"
    _name_ = "lipschitz_optimizer"
    __name__ = "LipschitzOptimizer"

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
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

    # =========================================================================
    # SMBO Template Methods
    # =========================================================================
    # Note: finish_initialization() and iterate() are inherited from SMBO.
    # LipschitzOptimizer only implements the algorithm-specific methods.

    def _training(self) -> None:
        """Prepare candidate positions for Lipschitz bound computation.

        Unlike traditional SMBO algorithms that train a surrogate model,
        Lipschitz optimization computes bounds analytically from the
        estimated Lipschitz constant. This method only prepares the
        candidate positions for evaluation.
        """
        self.pos_comb = self._sampling(self.all_pos_comb)

    def _expected_improvement(self) -> np.ndarray:
        """Compute Lipschitz upper bounds for candidate positions.

        The upper bound at each candidate position is computed using:
            upper_bound = min(y_i + L * dist(x, x_i)) for all observed (x_i, y_i)

        where L is the estimated Lipschitz constant.

        Returns
        -------
        np.ndarray
            1D array of upper bounds, one per candidate position.
            Higher values indicate more potential for improvement.
        """
        lip_func = LipschitzFunction(self.pos_comb)
        upper_bound_l = lip_func.calculate(
            self.X_sample, self.Y_sample, self.score_best
        )
        # Flatten from (n, 1) to (n,) for SMBO template compatibility
        return upper_bound_l.flatten()

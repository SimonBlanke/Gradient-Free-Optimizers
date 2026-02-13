# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Forest (Tree Ensemble) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .acquisition_function import ExpectedImprovement
from .smbo import SMBO
from .surrogate_models import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

if TYPE_CHECKING:
    import pandas as pd


def normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.array(arr)
    arr_min = arr.min()
    arr_max = arr.max()
    range_ = arr_max - arr_min

    if range_ == 0:
        return np.random.uniform(0, 1, size=arr.shape)
    else:
        return (arr - arr_min) / range_


tree_regressor_dict = {
    "random_forest": RandomForestRegressor,
    "extra_tree": ExtraTreesRegressor,
    "gradient_boost": GradientBoostingRegressor,
}


class ForestOptimizer(SMBO):
    """Sequential model-based optimization using tree ensemble surrogates.

    Dimension Support:
        - Continuous: YES (tree ensembles handle continuous naturally)
        - Categorical: YES (with index encoding)
        - Discrete: YES (tree ensembles handle discrete naturally)

    Uses a tree-based ensemble (Random Forest, Extra Trees, or Gradient Boosting)
    as surrogate model. Tree ensembles can capture non-linear relationships and
    provide uncertainty estimates via prediction variance across trees.

    The key advantage of tree ensembles over GPs is that they scale better to
    large datasets and high-dimensional spaces, though they may be less
    sample-efficient on smooth functions.

    Based on the forest-optimizer in the scikit-optimize package.

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
        Probability of random iteration.
    nth_process : int, optional
        Process index for parallel optimization.
    warm_start_smbo : pd.DataFrame, optional
        Previous results to initialize the model.
    max_sample_size : int, default=10000000
        Maximum positions to consider.
    sampling : dict or False, default=None
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    tree_regressor : str, default="extra_tree"
        Type of tree ensemble: "random_forest", "extra_tree", or "gradient_boost".
    tree_para : dict, default={"n_estimators": 100}
        Parameters passed to the tree regressor.
    xi : float, default=0.03
        Exploration-exploitation parameter for Expected Improvement.
    """

    name = "Forest Optimization"
    _name_ = "forest_optimization"
    __name__ = "ForestOptimizer"

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
        tree_regressor: Literal[
            "random_forest", "extra_tree", "gradient_boost"
        ] = "extra_tree",
        tree_para: dict[str, Any] | None = None,
        xi: float = 0.03,
    ) -> None:
        if tree_para is None:
            tree_para = {"n_estimators": 100}

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

        self.tree_regressor = tree_regressor
        self.tree_para = tree_para
        self.regr = tree_regressor_dict[tree_regressor](**self.tree_para)
        self.xi = xi

    def _expected_improvement(self) -> np.ndarray:
        """Compute Expected Improvement for all candidate positions.

        Uses the tree ensemble's prediction variance to estimate uncertainty.

        Returns
        -------
        np.ndarray
            Acquisition values for each candidate position.
        """
        self.pos_comb = self._sampling(self.all_pos_comb)

        acqu_func = ExpectedImprovement(self.regr, self.pos_comb, self.xi)
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _training(self) -> None:
        """Fit the tree ensemble on training data."""
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        if len(Y_sample) == 0:
            return

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gradient_free_optimizers._init_utils import (
    get_default_initialize,
    get_default_sampling,
)

from ..base_optimizer import BaseOptimizer
from ..core_optimizer.converter import ArrayLike
from .sampling import InitialSampler

if TYPE_CHECKING:
    import pandas as pd

np.seterr(divide="ignore", invalid="ignore")
from gradient_free_optimizers._array_backend import (
    HAS_NUMPY,
    array,
    isinf,
    isnan,
    meshgrid,
)
from gradient_free_optimizers._array_backend import (
    random as np_random,
)

# Import numpy only for warm_start pandas operations
if HAS_NUMPY:
    import numpy as np


class SMBO(BaseOptimizer):
    """Base class for Sequential Model-Based Optimization algorithms.

    SMBO algorithms use a surrogate model to approximate the objective function
    and an acquisition function to decide which positions to evaluate next.
    The surrogate model is trained on evaluated positions and their scores,
    then used to predict promising candidates.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict or None, default=None
        Strategy for generating initial positions before model-based search.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
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
        If None, uses {"random": 1000000}.
    replacement : bool, default=True
        Whether to allow re-evaluation of the same position.

    Attributes
    ----------
    X_sample : list
        List of evaluated positions (training data for surrogate).
    Y_sample : list
        List of scores corresponding to X_sample.
    sampler : InitialSampler
        Sampler for generating candidate positions.
    """

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
        if initialize is None:
            initialize = get_default_initialize()
        if sampling is None:
            sampling = get_default_sampling()

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
        self.sampling = sampling
        self.replacement = replacement

        self.sampler = InitialSampler(self.conv, max_sample_size)

        self.init_warm_start_smbo(warm_start_smbo)

    def init_warm_start_smbo(self, search_data: pd.DataFrame | None) -> None:
        """Initialize X_sample and Y_sample from previous optimization data.

        Filters out invalid values (NaN, inf) and values outside the search space.

        Parameters
        ----------
        search_data : pd.DataFrame or None
            DataFrame with parameter columns and a 'score' column.
        """
        if search_data is not None:
            # Note: warm_start uses pandas DataFrame, which requires numpy
            # This is expected since users providing warm_start data have dependencies
            import numpy as np

            # filter out nan and inf
            warm_start_smbo = search_data[
                ~search_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)
            ]

            # filter out elements that are not in search space
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

    @staticmethod
    def track_X_sample(iterate: Callable) -> Callable:
        """Decorator that appends returned position to X_sample."""

        def wrapper(self, *args: Any, **kwargs: Any) -> ArrayLike:
            pos = iterate(self, *args, **kwargs)
            self.X_sample.append(pos)
            return pos

        return wrapper

    @staticmethod
    def track_y_sample(evaluate: Callable) -> Callable:
        """Decorator that appends score to Y_sample, skipping invalid scores."""

        def wrapper(self, score: float) -> None:
            evaluate(self, score)

            if math.isnan(score) if isinstance(score, float) else isnan(score):
                del self.X_sample[-1]
            elif math.isinf(score) if isinstance(score, float) else isinf(score):
                del self.X_sample[-1]
            else:
                self.Y_sample.append(score)

        return wrapper

    def _sampling(self, all_pos_comb: np.ndarray) -> np.ndarray:
        if self.sampling is False:
            return all_pos_comb
        elif "random" in self.sampling:
            return self.random_sampling(all_pos_comb)

    def random_sampling(self, pos_comb: np.ndarray) -> np.ndarray:
        n_samples = self.sampling["random"]
        n_pos_comb = (
            len(pos_comb) if hasattr(pos_comb, "__len__") else pos_comb.shape[0]
        )

        if n_pos_comb <= n_samples:
            return pos_comb
        else:
            _idx_sample = np_random.choice(n_pos_comb, n_samples, replace=False)
            # Handle both numpy arrays and GFOArray
            if hasattr(pos_comb, "shape"):
                pos_comb_sampled = pos_comb[_idx_sample, :]
            else:
                pos_comb_sampled = array([pos_comb[i] for i in _idx_sample])
            return pos_comb_sampled

    def _all_possible_pos(self) -> np.ndarray:
        pos_space = self.sampler.get_pos_space()
        n_dim = len(pos_space)

        # Create meshgrid and reshape
        grids = meshgrid(*pos_space)
        # Transpose and reshape to get all combinations
        all_pos_comb = array(grids).T.reshape(-1, n_dim)

        all_pos_comb_constr = []
        for pos in all_pos_comb:
            if self.conv.not_in_constraint(pos):
                all_pos_comb_constr.append(pos)

        all_pos_comb_constr = array(all_pos_comb_constr)
        return all_pos_comb_constr

    def memory_warning(self, max_sample_size: int) -> None:
        if (
            self.conv.search_space_size > self.warnings
            and max_sample_size > self.warnings
        ):
            warning_message0 = "\n Warning:"
            warning_message1 = (
                "\n search space size of "
                + str(self.conv.search_space_size)
                + " exceeding recommended limit."
            )
            warning_message3 = "\n Reduce search space size for better performance."
            logging.warning(warning_message0 + warning_message1 + warning_message3)

    @track_X_sample
    def init_pos(self) -> ArrayLike:
        return super().init_pos()

    @BaseOptimizer.track_new_pos
    @track_X_sample
    def iterate(self) -> ArrayLike:
        """Generate next position using surrogate model and acquisition function."""
        return self._propose_location()

    def _remove_position(self, position: ArrayLike) -> None:
        mask = np.all(self.all_pos_comb == position, axis=1)
        self.all_pos_comb = self.all_pos_comb[np.invert(mask)]

    @BaseOptimizer.track_new_score
    @track_y_sample
    def evaluate(self, score_new: float) -> None:
        """Evaluate the current position and update surrogate training data."""
        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

        if not self.replacement:
            self._remove_position(self.pos_new)

    @BaseOptimizer.track_new_score
    @track_y_sample
    def evaluate_init(self, score_new: float) -> None:
        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

    def _propose_location(self) -> ArrayLike:
        try:
            self._training()
        except ValueError:
            logging.warning(
                "Training sequential model failed. Performing random iteration."
            )
            return self.move_random_typed()

        exp_imp = self._expected_improvement()

        index_best = list(exp_imp.argsort()[::-1])
        all_pos_comb_sorted = self.pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

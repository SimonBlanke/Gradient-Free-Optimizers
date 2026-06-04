# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Bayesian Optimization with Gaussian Process."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from gradient_free_optimizers._array_backend import (
    array,
    linalg,
    ndarray,
    random,
)

from .acquisition_function import (
    create_acquisition_function,
    normalize_acquisition_function_name,
)
from .smbo import SMBO
from .surrogate_models import GPR

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = array(arr)
    arr_min = arr.min()
    arr_max = arr.max()
    range_ = arr_max - arr_min

    if range_ == 0:
        return random.uniform(0, 1, size=arr.shape)
    else:
        return (arr - arr_min) / range_


class BayesianOptimizer(SMBO):
    """Bayesian Optimization with Gaussian Process surrogate.

    Dimension Support:
        - Continuous: YES (native GP support)
        - Categorical: YES (with index encoding)
        - Discrete: YES (treated as continuous, then rounded)

    Uses a Gaussian Process as surrogate model to approximate the objective
    function and Expected Improvement as acquisition function. The GP provides
    both mean predictions and uncertainty estimates, enabling principled
    exploration-exploitation trade-offs.

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
        Previous results to initialize the GP.
    max_sample_size : int, default=10000000
        Maximum positions to consider.
    sampling : dict or False, default=None
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    gpr : object, default=None
        Gaussian Process regressor instance. If None, uses default GPR.
    xi : float, default=0.03
        Exploration-exploitation parameter for Expected Improvement and
        Probability of Improvement. Higher values favor exploration.
    acquisition_function : str, default="expected_improvement"
        Acquisition function used to score candidate positions. Supports
        "expected_improvement", "probability_of_improvement", and
        "thompson_sampling". Short aliases "ei", "pi", and "thompson" are
        also accepted.
    strategy : object, optional
        Candidate strategy with ``filter_candidates`` and ``on_evaluation``
        methods. For example, ``TuRBO`` restricts candidate scoring to a
        trust region around the current best position.
    """

    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"
    __name__ = "BayesianOptimizer"

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
        boundary: str = "clip",
        warm_start_smbo: pd.DataFrame | None = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] | Literal[False] | None = None,
        replacement: bool = True,
        gpr=None,
        xi: float = 0.03,
        acquisition_function: str = "expected_improvement",
        strategy: Any | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            boundary=boundary,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

        # Instantiate GPR - supports both class and instance
        if gpr is None:
            self.gpr = GPR()
        elif isinstance(gpr, type):
            # User passed a class, instantiate it
            self.gpr = gpr()
        else:
            # User passed an instance
            self.gpr = gpr

        self.regr = self.gpr
        self.xi = xi
        self.acquisition_function = normalize_acquisition_function_name(
            acquisition_function
        )
        self.strategy = strategy
        self._validate_strategy(strategy)
        self._strategy_empty_filter_warned = False

        max_pos = self.conv.max_positions
        n_dims = len(max_pos)
        offsets = [0.0] * n_dims
        denoms = [1.0] * n_dims

        cont_idx = 0
        for i in range(n_dims):
            if self._continuous_mask is not None and self._continuous_mask[i]:
                bounds = self._continuous_bounds[cont_idx]
                offsets[i] = float(bounds[0])
                range_ = float(bounds[1]) - float(bounds[0])
                denoms[i] = range_ if range_ > 0 else 1.0
                cont_idx += 1
            else:
                denoms[i] = max(float(max_pos[i]), 1.0)

        self._x_norm_offset = array(offsets)
        self._x_norm_denom = array(denoms)

    @staticmethod
    def _validate_strategy(strategy) -> None:
        """Validate the optional Bayesian optimizer strategy object."""
        if strategy is None:
            return

        required_methods = ("filter_candidates", "on_evaluation")
        missing_methods = [
            method
            for method in required_methods
            if not callable(getattr(strategy, method, None))
        ]
        if missing_methods:
            missing = ", ".join(missing_methods)
            raise ValueError(
                "strategy must provide callable methods: "
                f"{', '.join(required_methods)}. Missing: {missing}."
            )

    def _normalize_X(self, X):
        """Normalize positions to [0, 1] per dimension."""
        return (array(X, dtype=float) - self._x_norm_offset) / self._x_norm_denom

    def _expected_improvement(self) -> ndarray:
        """Compute acquisition values for all candidate positions."""
        self.pos_comb = self._sampling(self.all_pos_comb)

        pos_comb_norm = self._normalize_X(self.pos_comb)
        self.pos_comb, pos_comb_norm = self._apply_strategy_candidate_filter(
            self.pos_comb, pos_comb_norm
        )

        acqu_func = create_acquisition_function(
            self.acquisition_function,
            self.regr,
            pos_comb_norm,
            self.xi,
            rng=self._rng_acquisition,
        )
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _apply_strategy_candidate_filter(self, pos_comb, pos_comb_norm):
        """Restrict candidate positions through the configured strategy."""
        if self.strategy is None:
            return pos_comb, pos_comb_norm

        center = self._pos_best if self._pos_best is not None else self._pos_current
        if center is None:
            return pos_comb, pos_comb_norm

        center_norm = self._normalize_X([center])[0]
        if (
            hasattr(self.strategy, "initialize")
            and hasattr(self.strategy, "initialized")
            and not self.strategy.initialized
        ):
            self.strategy.initialize(center_norm)

        bounds_low = array([0.0] * len(center_norm))
        bounds_high = array([1.0] * len(center_norm))
        mask = self.strategy.filter_candidates(
            pos_comb_norm,
            center_norm,
            bounds_low,
            bounds_high,
        )

        filtered_pos_comb = pos_comb[mask]
        if len(filtered_pos_comb) == 0:
            if not self._strategy_empty_filter_warned:
                logger.warning(
                    "Bayesian optimizer strategy filtered out all candidates. "
                    "Using the unfiltered candidate set for this iteration."
                )
                self._strategy_empty_filter_warned = True
            return pos_comb, pos_comb_norm

        return filtered_pos_comb, pos_comb_norm[mask]

    def _training(self) -> None:
        """Fit the Gaussian Process on normalized training data."""
        X_sample = self._normalize_X(array(self.X_sample))
        Y_sample = array(self.Y_sample)

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

    def _iterate_batch(self, n):
        """Train GP once and select n diverse positions."""
        try:
            self._training()
            exp_imp = self._expected_improvement()
            positions = self._select_diverse_batch(exp_imp, n)
            while len(positions) < n:
                positions.append(self._move_random())
            return [self._clip_position(pos) for pos in positions]
        except (ValueError, linalg.LinAlgError):
            return [self._clip_position(self._move_random()) for _ in range(n)]

    def _evaluate_batch(self, positions, scores):
        """Process batch results through the standard evaluate chain."""
        for pos, score in zip(positions, scores):
            self._pos_new = pos
            self._evaluate(score)

    def _on_evaluate(self, score_new: float) -> None:
        """Update SMBO state and notify the configured strategy."""
        score_best_before = self._score_best
        improved = score_new > score_best_before

        super()._on_evaluate(score_new)

        if self.strategy is not None:
            self.strategy.on_evaluation(score_new, improved)

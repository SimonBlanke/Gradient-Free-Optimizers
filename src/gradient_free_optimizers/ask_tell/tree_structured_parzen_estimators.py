# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Tree-structured Parzen Estimator optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_sampling
from ..optimizers import (
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
)


class TreeStructuredParzenEstimators(_TreeStructuredParzenEstimators, AskTell):
    """Tree-structured Parzen Estimator optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initial_evaluations : list[tuple[dict, float]]
        Previously evaluated parameters and their scores to seed the optimizer.
    constraints : list, optional
        Constraint functions restricting the search space.
    random_state : int or None, default=None
        Seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart.
    warm_start_smbo : object or None, default=None
        Previous SMBO state for warm-starting the surrogate model.
    max_sample_size : int, default=10000000
        Maximum candidate points for acquisition optimization.
    sampling : dict, default={"random": 1000000}
        Candidate sampling configuration.
    replacement : bool, default=True
        Whether to sample candidates with replacement.
    gamma_tpe : float, default=0.2
        Quantile threshold for splitting observations into good and bad groups.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        warm_start_smbo: object | None = None,
        max_sample_size: int = 10000000,
        sampling: dict[Literal["random"], int] = None,
        replacement: bool = True,
        gamma_tpe: float = 0.2,
    ):
        if constraints is None:
            constraints = []
        if sampling is None:
            sampling = get_default_sampling()

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
            gamma_tpe=gamma_tpe,
        )

        self._process_initial_evaluations(initial_evaluations)

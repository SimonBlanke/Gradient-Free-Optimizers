# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""DIRECT algorithm optimizer with ask/tell interface."""

from typing import Literal

import pandas as pd

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import DirectAlgorithm as _DirectAlgorithm


class DirectAlgorithm(_DirectAlgorithm, AskTell):
    """DIRECT algorithm optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        Constraint functions restricting the search space.
    random_state : int or None, default=None
        Seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart.
    warm_start : pd.DataFrame or None, default=None
        Previous optimization results to warm-start the algorithm.
    resolution : int, default=100
        Number of grid points for continuous dimensions.
    warm_start_smbo : pd.DataFrame or None, default=None
        Legacy parameter, kept for backwards compatibility.
    max_sample_size : int, default=10000000
        Legacy parameter, kept for backwards compatibility.
    sampling : dict or None, default=None
        Legacy parameter, kept for backwards compatibility.
    replacement : bool, default=True
        Legacy parameter, kept for backwards compatibility.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = None,
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        warm_start: pd.DataFrame = None,
        resolution: int = 100,
        warm_start_smbo: pd.DataFrame = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] = None,
        replacement: bool = True,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            warm_start=warm_start,
            resolution=resolution,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

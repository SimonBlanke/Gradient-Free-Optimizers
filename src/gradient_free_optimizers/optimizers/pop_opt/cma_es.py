# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_population_optimizer import BasePopulationOptimizer

if TYPE_CHECKING:
    pass


class CMAESOptimizer(BasePopulationOptimizer):
    name = "CMA-ES"
    _name_ = "cma_es"
    __name__ = "CMAESOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        population: int | None = None,
        mu: int | None = None,
        sigma: float = 0.3,
        ipop_restart: bool = False,
    ) -> None:
        n = len(search_space)

        if population is None:
            population = 4 + int(3 * np.log(n)) if n > 0 else 4

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Differential evolution optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import (
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer,
)


class DifferentialEvolutionOptimizer(_DifferentialEvolutionOptimizer, AskTell):
    """Differential Evolution optimizer with ask/tell interface.

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
    population : int, default=10
        Number of individuals in the population.
    mutation_rate : float, default=0.9
        Scaling factor for the differential mutation vector.
    crossover_rate : float, default=0.9
        Probability that each parameter inherits from the mutant vector.
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
        population: int = 10,
        mutation_rate: float = 0.9,
        crossover_rate: float = 0.9,
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
            population=population,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

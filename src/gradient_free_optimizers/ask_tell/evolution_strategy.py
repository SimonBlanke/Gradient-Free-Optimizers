# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Evolution strategy optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import (
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
)


class EvolutionStrategyOptimizer(_EvolutionStrategyOptimizer, AskTell):
    """Evolution Strategy optimizer with ask/tell interface.

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
        Number of parent individuals in the population.
    offspring : int, default=20
        Number of offspring generated each generation.
    replace_parents : bool, default=False
        Whether only offspring compete for selection (comma strategy).
    mutation_rate : float, default=0.7
        Probability of mutating each parameter in an offspring.
    crossover_rate : float, default=0.3
        Probability of applying recombination to create offspring.
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
        offspring: int = 20,
        replace_parents: bool = False,
        mutation_rate: float = 0.7,
        crossover_rate: float = 0.3,
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
            offspring=offspring,
            replace_parents=replace_parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

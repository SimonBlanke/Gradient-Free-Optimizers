# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Evolution strategy optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import (
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
)


class EvolutionStrategyOptimizer(_EvolutionStrategyOptimizer, AskTell):
    """Evolution Strategy optimizer with ask/tell interface.

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
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = 10,
        offspring: int = 20,
        replace_parents: bool = False,
        mutation_rate: float = 0.7,
        crossover_rate: float = 0.3,
    ):
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            population=population,
            offspring=offspring,
            replace_parents=replace_parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

        self._process_initial_evaluations(initial_evaluations)

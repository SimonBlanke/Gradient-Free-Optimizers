# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Differential evolution optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import (
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer,
)


class DifferentialEvolutionOptimizer(_DifferentialEvolutionOptimizer, AskTell):
    """Differential Evolution optimizer with ask/tell interface.

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
        Number of individuals in the population.
    mutation_rate : float, default=0.9
        Scaling factor for the differential mutation vector.
    crossover_rate : float, default=0.9
        Probability that each parameter inherits from the mutant vector.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = 10,
        mutation_rate: float = 0.9,
        crossover_rate: float = 0.9,
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
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

        self._process_initial_evaluations(initial_evaluations)

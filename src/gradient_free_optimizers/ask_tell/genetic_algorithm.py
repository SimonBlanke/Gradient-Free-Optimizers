# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Genetic algorithm optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer


class GeneticAlgorithmOptimizer(_GeneticAlgorithmOptimizer, AskTell):
    """Genetic Algorithm optimizer with ask/tell interface.

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
    offspring : int, default=10
        Number of offspring to generate each generation.
    crossover : str, default="discrete-recombination"
        The crossover operator for combining parent genes.
    n_parents : int, default=2
        Number of parents selected for each crossover operation.
    mutation_rate : float, default=0.5
        Probability of mutating each gene in an offspring.
    crossover_rate : float, default=0.5
        Probability of applying crossover to create an offspring.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        conditions: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = 10,
        offspring: int = 10,
        crossover: str = "discrete-recombination",
        n_parents: int = 2,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.5,
    ):
        if constraints is None:
            constraints = []
        if conditions is None:
            conditions = []

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            conditions=conditions,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            population=population,
            offspring=offspring,
            crossover=crossover,
            n_parents=n_parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

        self._process_initial_evaluations(initial_evaluations)

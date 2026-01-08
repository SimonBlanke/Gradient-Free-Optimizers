# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import random
import numpy as np
from typing import Any, Callable

from gradient_free_optimizers._array_backend import random as np_random

from ._evolutionary_algorithm import EvolutionaryAlgorithmOptimizer
from ._individual import Individual
from ..core_optimizer.converter import ArrayLike

# Selection parameters for genetic algorithm
# Fraction of population selected as parents for crossover
FITTEST_PARENTS_FRACTION = 0.5
# Probability of replacing a fit parent with a random individual (diversity injection)
DIVERSITY_INJECTION_PROB = 0.01


class GeneticAlgorithmOptimizer(EvolutionaryAlgorithmOptimizer):
    """Genetic Algorithm inspired by biological evolution.

    Uses selection, crossover, and mutation operations to evolve a
    population of solutions. Fitter individuals are more likely to
    reproduce and pass their traits to offspring.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default={"grid": 4, "random": 2, "vertices": 4}
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    population : int, default=10
        Number of individuals in the population.
    offspring : int, default=10
        Number of offspring to generate per crossover batch.
    crossover : str, default="discrete-recombination"
        Crossover method to use.
    n_parents : int, default=2
        Number of parents for crossover.
    mutation_rate : float, default=0.5
        Probability of mutation operation.
    crossover_rate : float, default=0.5
        Probability of crossover operation.
    """

    name = "Genetic Algorithm"
    _name_ = "genetic_algorithm"
    __name__ = "GeneticAlgorithmOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        population: int = 10,
        offspring: int = 10,
        crossover: str = "discrete-recombination",
        n_parents: int = 2,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.5,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.population = population
        self.offspring = offspring
        self.crossover = crossover
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        self.offspring_l = []

    def fittest_parents(self) -> list[Individual]:
        self.sort_pop_best_score()

        n_fittest = int(len(self.pop_sorted) * FITTEST_PARENTS_FRACTION)

        best_l = self.pop_sorted[:n_fittest]
        worst_l = self.pop_sorted[n_fittest:]

        if DIVERSITY_INJECTION_PROB >= random.random():
            best_l[random.randint(0, len(best_l) - 1)] = random.choice(worst_l)

        return best_l

    def _crossover(self) -> None:
        fittest_parents = self.fittest_parents()
        selected_parents = random.sample(fittest_parents, self.n_parents)

        for _ in range(self.offspring):
            parent_pos_l = [parent.pos_new for parent in selected_parents]
            offspring = self.discrete_recombination(parent_pos_l)
            offspring = self._constraint_loop(offspring)
            self.offspring_l.append(offspring)

    def _constraint_loop(self, position: ArrayLike) -> ArrayLike:
        while True:
            if self.conv.not_in_constraint(position):
                return position
            position = self.p_current.move_climb(position, epsilon_mod=0.3)

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def init_pos(self) -> ArrayLike:
        nth_pop = self.nth_trial % len(self.individuals)
        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def iterate(self) -> ArrayLike:
        """Generate next position via mutation or crossover."""
        n_ind = len(self.individuals)

        if n_ind == 1:
            self.p_current = self.individuals[0]
            return self.p_current.iterate()

        self.sort_pop_best_score()
        rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[rnd_int]

        total_rate = self.mutation_rate + self.crossover_rate
        rand = np_random.uniform(low=0, high=total_rate)

        if rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            if not self.offspring_l:
                self._crossover()
            self.p_current.pos_new = self.offspring_l.pop(0)
            return self.p_current.pos_new

    @EvolutionaryAlgorithmOptimizer.track_new_score
    def evaluate(self, score_new: float) -> None:
        self.p_current.evaluate(score_new)

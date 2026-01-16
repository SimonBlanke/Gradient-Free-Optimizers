# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from gradient_free_optimizers._array_backend import random as np_random
from gradient_free_optimizers._init_utils import get_default_initialize

from ._evolutionary_algorithm import EvolutionaryAlgorithmOptimizer
from ._individual import Individual


class EvolutionStrategyOptimizer(EvolutionaryAlgorithmOptimizer):
    """Evolution Strategy optimization algorithm.

    Uses mutation and recombination to evolve a population. Can operate
    in (mu, lambda) mode where parents are replaced, or (mu + lambda)
    mode where parents compete with offspring.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    population : int, default=10
        Number of parent individuals (mu).
    offspring : int, default=20
        Number of offspring to generate (lambda).
    replace_parents : bool, default=False
        If True, use (mu, lambda); if False, use (mu + lambda).
    mutation_rate : float, default=0.7
        Probability of mutation operation.
    crossover_rate : float, default=0.3
        Probability of crossover operation.
    """

    name = "Evolution Strategy"
    _name_ = "evolution_strategy"
    __name__ = "EvolutionStrategyOptimizer"

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
        population: int = 10,
        offspring: int = 20,
        replace_parents: bool = False,
        mutation_rate: float = 0.7,
        crossover_rate: float = 0.3,
    ) -> None:
        if initialize is None:
            initialize = get_default_initialize()

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
        self.replace_parents = replace_parents
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

    def _cross(self):
        while True:
            if len(self.individuals) > 2:
                rnd_int2 = random.choice(
                    [i for i in range(0, self.n_ind - 1) if i not in [self.rnd_int]]
                )
            else:
                rnd_int2 = random.choice(
                    [i for i in range(0, self.n_ind) if i not in [self.rnd_int]]
                )

            p_sec = self.pop_sorted[rnd_int2]
            p_worst = self.pop_sorted[-1]

            two_best_pos = [self.p_current.pos_current, p_sec.pos_current]
            pos_new = self.discrete_recombination(two_best_pos)

            self.p_current = p_worst
            p_worst.pos_new = pos_new

            if self.conv.not_in_constraint(pos_new):
                return pos_new

            return self.p_current.move_climb_typed(pos_new)

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def iterate(self):
        """Generate next position via mutation or recombination."""
        self.n_ind = len(self.individuals)

        if self.n_ind == 1:
            self.p_current = self.individuals[0]
            return self.p_current.iterate()

        self.sort_pop_best_score()
        self.rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[self.rnd_int]

        total_rate = self.mutation_rate + self.crossover_rate
        rand = np_random.uniform(low=0, high=total_rate)

        if rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            return self._cross()

    @EvolutionaryAlgorithmOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)

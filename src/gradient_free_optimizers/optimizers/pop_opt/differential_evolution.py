# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from gradient_free_optimizers._array_backend import array

from ..core_optimizer.converter import ArrayLike
from ._evolutionary_algorithm import EvolutionaryAlgorithmOptimizer
from ._individual import Individual


class DifferentialEvolutionOptimizer(EvolutionaryAlgorithmOptimizer):
    """Differential Evolution optimization algorithm.

    Evolves a population using vector differences between randomly selected
    individuals. The mutation creates donor vectors that are combined with
    target vectors through crossover.

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
    mutation_rate : float, default=0.9
        Scaling factor F for difference vectors.
    crossover_rate : float, default=0.9
        Probability of gene exchange in crossover.
    """

    name = "Differential Evolution"
    _name_ = "differential_evolution"
    __name__ = "DifferentialEvolutionOptimizer"

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
        mutation_rate: float = 0.9,
        crossover_rate: float = 0.9,
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
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        self.offspring_l = []

    def mutation(self, f: float = 1) -> ArrayLike:
        ind_selected = random.sample(self.individuals, 3)

        x_1, x_2, x_3 = (ind.pos_best for ind in ind_selected)
        return array(x_1) + self.mutation_rate * (array(x_2) - array(x_3))

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
        """Generate trial vector via mutation and crossover."""
        self.p_current = self.individuals[self.nth_trial % len(self.individuals)]
        target_vector = self.p_current.pos_new

        mutant_vector = self.mutation()

        crossover_rates = [1 - self.crossover_rate, self.crossover_rate]
        pos_new = self.discrete_recombination(
            [target_vector, mutant_vector],
            crossover_rates,
        )
        pos_new = self.conv2pos(pos_new)
        pos_new = self._constraint_loop(pos_new)

        self.p_current.pos_new = self.conv2pos(pos_new)
        return self.p_current.pos_new

    @EvolutionaryAlgorithmOptimizer.track_new_score
    def evaluate(self, score_new: float) -> None:
        self.p_current.evaluate(score_new)  # selection

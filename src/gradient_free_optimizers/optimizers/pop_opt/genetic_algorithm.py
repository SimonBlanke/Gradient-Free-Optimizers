# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ._evolutionary_algorithm import EvolutionaryAlgorithmOptimizer
from ._individual import Individual


class GeneticAlgorithmOptimizer(EvolutionaryAlgorithmOptimizer):
    name = "Genetic Algorithm"
    _name_ = "genetic_algorithm"
    __name__ = "GeneticAlgorithmOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=10,
        offspring=10,
        crossover="discrete-recombination",
        n_parents=2,
        mutation_rate=0.5,
        crossover_rate=0.5,
    ):
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

    def fittest_parents(self):
        fittest_parents_f = 0.5

        self.sort_pop_best_score()

        n_fittest = int(len(self.pop_sorted) * fittest_parents_f)

        best_l = self.pop_sorted[:n_fittest]
        worst_l = self.pop_sorted[n_fittest:]

        if 0.01 >= random.random():
            best_l[random.randint(0, len(best_l) - 1)] = random.choice(worst_l)

        return best_l

    def _crossover(self):
        fittest_parents = self.fittest_parents()
        selected_parents = random.sample(fittest_parents, self.n_parents)

        for _ in range(self.offspring):
            parent_pos_l = [parent.pos_new for parent in selected_parents]
            offspring = self.discrete_recombination(parent_pos_l)
            offspring = self._constraint_loop(offspring)
            self.offspring_l.append(offspring)

    def _constraint_loop(self, position):
        while True:
            if self.conv.not_in_constraint(position):
                return position
            position = self.p_current.move_climb(position, epsilon_mod=0.3)

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)
        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def iterate(self):
        n_ind = len(self.individuals)

        if n_ind == 1:
            self.p_current = self.individuals[0]
            return self.p_current.iterate()

        self.sort_pop_best_score()
        rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[rnd_int]

        total_rate = self.mutation_rate + self.crossover_rate
        rand = np.random.uniform(low=0, high=total_rate)

        if rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            if not self.offspring_l:
                self._crossover()
            self.p_current.pos_new = self.offspring_l.pop(0)
            return self.p_current.pos_new

    @EvolutionaryAlgorithmOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)

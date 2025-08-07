# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from ._evolutionary_algorithm import EvolutionaryAlgorithmOptimizer
from ._individual import Individual


class EvolutionStrategyOptimizer(EvolutionaryAlgorithmOptimizer):
    name = "Evolution Strategy"
    _name_ = "evolution_strategy"
    __name__ = "EvolutionStrategyOptimizer"

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
        offspring=20,
        replace_parents=False,
        mutation_rate=0.7,
        crossover_rate=0.3,
        self_adaptation=True,
        adaptation_window=10,
        adaptation_factor=1.22,
        target_success_rate=0.2,
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
        self.replace_parents = replace_parents
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Self-adaptation parameters
        self.self_adaptation = self_adaptation
        self.adaptation_window = adaptation_window
        self.adaptation_factor = adaptation_factor
        self.target_success_rate = target_success_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

    def _create_population(self, Optimizer):
        """Override to pass self-adaptation parameters to Individual instances."""
        if isinstance(self.population, int):
            pop_size = self.population
        else:
            pop_size = len(self.population)
        diff_init = pop_size - self.init.n_inits

        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

        if isinstance(self.population, int):
            from .base_population_optimizer import split

            distributed_init_positions = split(
                self.init.init_positions_l, self.population
            )

            population = []
            for init_positions in distributed_init_positions:
                init_values = self.conv.positions2values(init_positions)
                init_paras = self.conv.values2paras(init_values)

                population.append(
                    Optimizer(
                        self.conv.search_space,
                        rand_rest_p=self.rand_rest_p,
                        initialize={"warm_start": init_paras},
                        constraints=self.constraints,
                        self_adaptation=self.self_adaptation,
                        adaptation_window=self.adaptation_window,
                        adaptation_factor=self.adaptation_factor,
                        target_success_rate=self.target_success_rate,
                    )
                )
        else:
            population = self.population

        return population

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

            return self.p_current.move_climb(pos_new)

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    @EvolutionaryAlgorithmOptimizer.track_new_pos
    def iterate(self):
        self.n_ind = len(self.individuals)

        if self.n_ind == 1:
            self.p_current = self.individuals[0]
            return self.p_current.iterate()

        self.sort_pop_best_score()
        self.rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[self.rnd_int]

        total_rate = self.mutation_rate + self.crossover_rate
        rand = np.random.uniform(low=0, high=total_rate)

        if rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            return self._cross()

    @EvolutionaryAlgorithmOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)

    def _get_population_adaptation_info(self):
        """Return adaptation info for all individuals in the population."""
        if not self.self_adaptation:
            return {"self_adaptation": False, "population_size": len(self.individuals)}

        adaptation_info = {
            "self_adaptation": True,
            "population_size": len(self.individuals),
            "individuals": [],
        }

        for i, individual in enumerate(self.individuals):
            individual_info = individual.get_adaptation_info()
            individual_info["individual_id"] = i
            adaptation_info["individuals"].append(individual_info)

        # Calculate population statistics
        if adaptation_info["individuals"]:
            mutation_strengths = [
                ind["mutation_strength"]
                for ind in adaptation_info["individuals"]
                if "mutation_strength" in ind
            ]
            success_rates = [
                ind["success_rate"]
                for ind in adaptation_info["individuals"]
                if "success_rate" in ind
            ]

            if mutation_strengths:
                adaptation_info["population_stats"] = {
                    "avg_mutation_strength": sum(mutation_strengths)
                    / len(mutation_strengths),
                    "min_mutation_strength": min(mutation_strengths),
                    "max_mutation_strength": max(mutation_strengths),
                    "avg_success_rate": (
                        sum(success_rates) / len(success_rates) if success_rates else 0
                    ),
                }

        return adaptation_info

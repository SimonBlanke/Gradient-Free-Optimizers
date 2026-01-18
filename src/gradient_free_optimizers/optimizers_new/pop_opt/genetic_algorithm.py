# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Genetic Algorithm (GA) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ._individual import Individual

if TYPE_CHECKING:
    pass

# Selection parameters for genetic algorithm
# Fraction of population selected as parents for crossover
FITTEST_PARENTS_FRACTION = 0.5
# Probability of replacing a fit parent with a random individual (diversity injection)
DIVERSITY_INJECTION_PROB = 0.01


class GeneticAlgorithmOptimizer(BasePopulationOptimizer):
    """Genetic Algorithm inspired by biological evolution.

    Dimension Support:
        - Continuous: YES (mutation with Gaussian noise)
        - Categorical: YES (random category mutation)
        - Discrete: YES (mutation with Gaussian noise, rounded)

    Uses selection, crossover, and mutation operations to evolve a
    population of solutions. Fitter individuals are more likely to
    reproduce and pass their traits to offspring.

    The algorithm operates as follows:
    1. Selection: Select fittest parents from population
    2. Crossover: Combine parent positions to create offspring
    3. Mutation: Apply small changes via hill climbing

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
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
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        population: int = 10,
        offspring: int = 10,
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
            population=population,
        )

        self.offspring = offspring
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Create population of individuals
        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        # Queue for offspring from crossover
        self.offspring_l = []

        # Initialize RNG for reproducibility
        self._rng = np.random.default_rng(self.random_seed)

    def discrete_recombination(self, parent_pos_l, crossover_rates=None):
        """Combine parent positions using discrete recombination.

        For each dimension, randomly select from one of the parents.
        This works for all dimension types (continuous, categorical, discrete).

        Parameters
        ----------
        parent_pos_l : list of np.ndarray
            List of parent positions.
        crossover_rates : list of float, optional
            Probability weights for each parent. If None, uniform selection.

        Returns
        -------
        np.ndarray
            Offspring position.
        """
        n_parents = len(parent_pos_l)
        size = len(parent_pos_l[0])

        # Select parent index for each position
        if crossover_rates is None:
            crossover_rates = [1.0 / n_parents] * n_parents

        choice = []
        for _ in range(size):
            choice.append(self._rng.choice(n_parents, p=crossover_rates))

        # Build result by selecting from parents
        result = []
        for i, parent_idx in enumerate(choice):
            result.append(parent_pos_l[parent_idx][i])

        return np.array(result)

    def fittest_parents(self) -> list[Individual]:
        """Select the fittest individuals as parents.

        Returns the top FITTEST_PARENTS_FRACTION of the population.
        With small probability, injects diversity by replacing one
        fit parent with a less fit individual.

        Returns
        -------
        list[Individual]
            List of selected parent individuals.
        """
        self.sort_pop_best_score()

        n_fittest = max(1, int(len(self.pop_sorted) * FITTEST_PARENTS_FRACTION))

        best_l = list(self.pop_sorted[:n_fittest])
        worst_l = list(self.pop_sorted[n_fittest:])

        # Diversity injection: occasionally replace a good parent with a random one
        if worst_l and random.random() < DIVERSITY_INJECTION_PROB:
            best_l[random.randint(0, len(best_l) - 1)] = random.choice(worst_l)

        return best_l

    def _crossover(self) -> None:
        """Generate offspring via crossover.

        Selects n_parents from the fittest individuals and creates
        offspring via discrete recombination. Offspring are stored
        in offspring_l queue for later use.
        """
        fittest_parents = self.fittest_parents()

        # Need at least n_parents to do crossover
        if len(fittest_parents) < self.n_parents:
            return

        selected_parents = random.sample(fittest_parents, self.n_parents)

        for _ in range(self.offspring):
            parent_pos_l = [parent.pos_current for parent in selected_parents]
            # Skip if any parent has None position
            if any(pos is None for pos in parent_pos_l):
                continue

            offspring = self.discrete_recombination(parent_pos_l)
            offspring = self._constraint_loop(offspring)
            self.offspring_l.append(offspring)

    def _constraint_loop(self, position: np.ndarray) -> np.ndarray:
        """Ensure position satisfies constraints.

        If position violates constraints, use hill climbing to find
        a valid position nearby.

        Parameters
        ----------
        position : np.ndarray
            Position to check.

        Returns
        -------
        np.ndarray
            Valid position.
        """
        max_tries = 100
        for _ in range(max_tries):
            if self.conv.not_in_constraint(position):
                return position
            position = self.p_current.move_climb_typed(position, epsilon_mod=0.3)
        return position

    def init_pos(self) -> np.ndarray:
        """Initialize current individual and return its starting position.

        Returns
        -------
        np.ndarray
            Initial position for the current individual.
        """
        nth_pop = self.nth_trial % len(self.individuals)
        self.p_current = self.individuals[nth_pop]

        # Get initial position from individual
        if self.p_current.nth_init < len(self.p_current.init.init_positions_l):
            pos = self.p_current.init_pos()
        else:
            # Fall back to random position
            pos = self.p_current.init.move_random_typed()
            self.p_current.pos_current = pos
            self.p_current.pos_new = pos
            self.p_current.pos_new_list.append(pos)

        # Check constraints
        if not self.conv.not_in_constraint(pos):
            max_tries = 100
            for _ in range(max_tries):
                pos = self.p_current.init.move_random_typed()
                if self.conv.not_in_constraint(pos):
                    break
            self.p_current.pos_current = pos
            self.p_current.pos_new = pos
            if self.p_current.pos_new_list:
                self.p_current.pos_new_list[-1] = pos

        # Track position on main optimizer
        self.pos_new = pos
        self.pos_new_list.append(pos)

        return pos

    def evaluate_init(self, score_new: float) -> None:
        """Evaluate during initialization phase.

        Tracks score on both main optimizer and current individual.
        """
        # Track on main optimizer (this increments nth_trial)
        self._track_score(score_new)

        # Track on current individual
        self.p_current.evaluate_init(score_new)

        # Update tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

    def iterate(self) -> np.ndarray:
        """Generate next position via mutation or crossover.

        With probability proportional to mutation_rate, apply mutation.
        Otherwise, use crossover to generate offspring.

        Returns
        -------
        np.ndarray
            New position for evaluation.
        """
        n_ind = len(self.individuals)

        # Single individual: just mutate
        if n_ind == 1:
            self.p_current = self.individuals[0]
            pos_new = self.p_current.iterate()
            self.pos_new = pos_new
            self.pos_new_list.append(pos_new)
            return pos_new

        # Select a random individual (weighted toward fitter ones)
        self.sort_pop_best_score()
        rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[rnd_int]

        # Decide: mutation or crossover
        total_rate = self.mutation_rate + self.crossover_rate
        rand = self._rng.uniform(0, total_rate)

        if rand <= self.mutation_rate:
            # Mutation: use individual's iterate (hill climbing)
            pos_count_before = len(self.p_current.pos_new_list)
            pos_new = self.p_current.iterate()

            # Check constraints
            if not self.conv.not_in_constraint(pos_new):
                # Restore and try random
                while len(self.p_current.pos_new_list) > pos_count_before:
                    self.p_current.pos_new_list.pop()

                max_tries = 100
                for _ in range(max_tries):
                    pos_new = self.p_current.init.move_random_typed()
                    if self.conv.not_in_constraint(pos_new):
                        break

                self.p_current.pos_new = pos_new
                self.p_current.pos_new_list.append(pos_new)
        else:
            # Crossover: get offspring from queue (generate if empty)
            if not self.offspring_l:
                self._crossover()

            if self.offspring_l:
                pos_new = self.offspring_l.pop(0)
                self.p_current.pos_new = pos_new
                self.p_current.pos_new_list.append(pos_new)
            else:
                # Fallback to mutation if crossover failed
                pos_new = self.p_current.iterate()

        # Track position on main optimizer
        self.pos_new = pos_new
        self.pos_new_list.append(pos_new)

        return pos_new

    def _evaluate(self, score_new: float) -> None:
        """Evaluate current individual.

        Delegates to the individual's evaluate method which handles
        personal best tracking and sigma adaptation.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Delegate to current individual
        self.p_current.evaluate(score_new)

        # Update global tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

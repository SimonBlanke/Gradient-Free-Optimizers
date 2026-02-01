# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Genetic Algorithm (GA) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL

Template Method Pattern Compliance:
    - Does NOT override iterate() - uses CoreOptimizer's orchestration
    - Implements _iterate_*_batch() for dimension-type-aware GA operations
    - Overrides init_pos()/evaluate_init() for population management (acceptable)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ._individual import Individual
from .base_population_optimizer import BasePopulationOptimizer

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

    Template Method Pattern:
        This optimizer follows the Template Method Pattern by implementing
        _iterate_*_batch() methods instead of overriding iterate().
        The GA operation (mutation or crossover) is computed once per iteration,
        then portions are extracted for each dimension type.

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
            population=population,
        )

        self.offspring = offspring
        self.crossover = crossover  # Currently only "discrete-recombination" supported
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

        # Iteration state for template method coordination
        self._iteration_setup_done = False
        self._ga_new_pos = None

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

    # =========================================================================
    # Population Initialization Hooks
    # =========================================================================

    def _init_pos(self, position) -> None:
        """Initialize current individual with the given position.

        Args:
            position: The initialization position from CoreOptimizer.init_pos()
        """
        # Select individual via round-robin (use nth_init-1 since it was incremented)
        nth_pop = (self.nth_init - 1) % len(self.individuals)
        self.p_current = self.individuals[nth_pop]

        # Track position on current individual
        self.p_current.pos_new = position.copy()
        self.p_current.pos_current = position.copy()

    def _evaluate_init(self, score_new: float) -> None:
        """Evaluate during initialization phase.

        Delegates evaluation to the current individual for individual-level tracking.

        Args:
            score_new: Score of the most recently evaluated init position
        """
        # Track on current individual
        self.p_current.score_new = score_new

        # Update individual's best if this is better
        if self.p_current.pos_best is None or score_new > self.p_current.score_best:
            self.p_current.pos_best = self.p_current.pos_new.copy()
            self.p_current.score_best = score_new

        # Update individual's current
        self.p_current.score_current = score_new

    # =========================================================================
    # Template Method Implementation - NO iterate() override!
    # =========================================================================

    def _setup_iteration(self) -> None:
        """Set up current iteration by selecting individual and computing position.

        Called lazily by the first _iterate_*_batch() method.
        Decides between mutation and crossover, computes the full position once,
        which is then extracted by the individual batch methods.
        """
        if self._iteration_setup_done:
            return

        n_ind = len(self.individuals)

        # Single individual: just mutate
        if n_ind == 1:
            self.p_current = self.individuals[0]
            self._ga_new_pos = self._compute_mutation_position()
            self._iteration_setup_done = True
            return

        # Select a random individual (weighted toward fitter ones)
        self.sort_pop_best_score()
        rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[rnd_int]

        # Decide: mutation or crossover
        total_rate = self.mutation_rate + self.crossover_rate
        rand = self._rng.uniform(0, total_rate)

        if rand <= self.mutation_rate:
            # Mutation: use individual's hill climbing
            self._ga_new_pos = self._compute_mutation_position()
        else:
            # Crossover: get offspring from queue (generate if empty)
            self._ga_new_pos = self._compute_crossover_position()

        self._iteration_setup_done = True

    def _compute_mutation_position(self) -> np.ndarray:
        """Compute new position via mutation using the individual's hill climbing.

        Uses the Individual's typed iteration which applies Gaussian noise
        to all dimension types appropriately.

        Returns
        -------
        np.ndarray
            New position after mutation.
        """
        # Random restart check
        if random.random() < self.rand_rest_p:
            return self.p_current.init.move_random_typed()

        # Guard against None positions during early iterations
        if self.p_current.pos_current is None:
            return self.p_current.init.move_random_typed()

        # Use individual's typed iteration (calls _iterate_*_batch methods)
        # The Individual inherits from HillClimbingOptimizer which has these implemented
        pos_new = self.p_current._iterate_typed(self.p_current.pos_current)

        # Check constraints - if violated, try to find a valid position
        if not self.conv.not_in_constraint(pos_new):
            max_tries = 100
            for _ in range(max_tries):
                pos_new = self.p_current.init.move_random_typed()
                if self.conv.not_in_constraint(pos_new):
                    break

        return pos_new

    def _compute_crossover_position(self) -> np.ndarray:
        """Compute new position via crossover from offspring queue.

        Uses discrete recombination to combine parent positions.
        Falls back to mutation if crossover fails.

        Returns
        -------
        np.ndarray
            New position from crossover or fallback mutation.
        """
        # Fill offspring queue if empty
        if not self.offspring_l:
            self._crossover()

        if self.offspring_l:
            return self.offspring_l.pop(0)
        else:
            # Fallback to mutation if crossover failed
            return self._compute_mutation_position()

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Generate continuous values using GA mutation/crossover.

        Returns the continuous portion of the GA-computed position.

        Returns
        -------
        np.ndarray
            New continuous values from GA operation.
        """
        self._setup_iteration()
        return self._ga_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Generate categorical indices using GA mutation/crossover.

        Returns the categorical portion of the GA-computed position.

        Returns
        -------
        np.ndarray
            New category indices from GA operation.
        """
        self._setup_iteration()
        return self._ga_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Generate discrete indices using GA mutation/crossover.

        Returns the discrete portion of the GA-computed position.

        Returns
        -------
        np.ndarray
            New discrete indices from GA operation.
        """
        self._setup_iteration()
        return self._ga_new_pos[self._discrete_mask]

    def _evaluate(self, score_new: float) -> None:
        """Evaluate current individual.

        Delegates to the individual's evaluate method which handles
        personal best tracking and sigma adaptation. Also resets
        iteration state for the next iteration.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Track position on individual (needed for personal best tracking)
        self.p_current.pos_new = self.pos_new

        # Delegate to current individual
        self.p_current.evaluate(score_new)

        # Update global tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._ga_new_pos = None

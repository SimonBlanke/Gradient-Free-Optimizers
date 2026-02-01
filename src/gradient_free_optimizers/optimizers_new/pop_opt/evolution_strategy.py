# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Evolution Strategy (ES) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL

Template Method Pattern Compliance:
    - Does NOT override iterate() - uses CoreOptimizer's orchestration
    - Implements _iterate_*_batch() for dimension-type-aware position generation
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


class EvolutionStrategyOptimizer(BasePopulationOptimizer):
    """Evolution Strategy optimization algorithm.

    Dimension Support:
        - Continuous: YES (adaptive Gaussian mutation)
        - Categorical: YES (probabilistic mutation)
        - Discrete: YES (adaptive Gaussian mutation, rounded)

    Uses mutation and recombination to evolve a population. Can operate
    in (mu, lambda) mode where parents are replaced, or (mu + lambda)
    mode where parents compete with offspring.

    The key difference from Genetic Algorithm is the emphasis on
    self-adaptive mutation step sizes (sigma) that evolve with the
    population.

    Template Method Pattern:
        This optimizer follows the Template Method Pattern by implementing
        _iterate_*_batch() methods instead of overriding iterate().
        The sub-optimizer's iterate() is called once per iteration, then
        portions are extracted for each dimension type.

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
        Number of parent individuals (mu).
    offspring : int, default=20
        Number of offspring to generate (lambda).
    mutation_rate : float, default=0.7
        Probability of mutation operation.
    crossover_rate : float, default=0.3
        Probability of crossover/recombination operation.
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
        self.replace_parents = replace_parents  # (mu,lambda) vs (mu+lambda)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Create population of individuals
        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        # Initialize RNG for reproducibility
        self._rng = np.random.default_rng(self.random_seed)

        # State for iterate
        self.rnd_int = 0
        self.n_ind = len(self.individuals)

        # Iteration state for template method coordination
        self._iteration_setup_done = False
        self._current_new_pos = None

    def discrete_recombination(self, parent_pos_l, crossover_rates=None):
        """Combine parent positions using discrete recombination.

        For each dimension, randomly select from one of the parents.

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

        if crossover_rates is None:
            crossover_rates = [1.0 / n_parents] * n_parents

        choice = []
        for _ in range(size):
            choice.append(self._rng.choice(n_parents, p=crossover_rates))

        result = []
        for i, parent_idx in enumerate(choice):
            result.append(parent_pos_l[parent_idx][i])

        return np.array(result)

    def _cross(self):
        """Perform recombination (crossover) operation.

        Selects a second parent different from the current one,
        performs discrete recombination, and assigns the result
        to the worst individual in the population.

        Returns
        -------
        np.ndarray
            New position from recombination.
        """
        # Select a second parent different from current
        if self.n_ind > 2:
            available = [i for i in range(self.n_ind - 1) if i != self.rnd_int]
        else:
            available = [i for i in range(self.n_ind) if i != self.rnd_int]

        if not available:
            # Fallback to mutation via individual's iterate
            return self.p_current.iterate()

        rnd_int2 = random.choice(available)

        p_sec = self.pop_sorted[rnd_int2]
        p_worst = self.pop_sorted[-1]

        # Guard against None positions
        if self.p_current.pos_current is None or p_sec.pos_current is None:
            return self.p_current.iterate()

        # Recombine the two parents
        two_best_pos = [self.p_current.pos_current, p_sec.pos_current]
        pos_new = self.discrete_recombination(two_best_pos)

        # Assign to worst individual
        self.p_current = p_worst
        p_worst.pos_new = pos_new  # Property setter auto-appends

        # Handle constraints
        if not self.conv.not_in_constraint(pos_new):
            pos_new = self.p_current.move_climb_typed(pos_new)
            self.p_current.pos_new = pos_new
            if self.p_current.pos_new_list:
                self.p_current.pos_new_list[-1] = pos_new

        return pos_new

    def init_pos(self) -> np.ndarray:
        """Initialize current individual and return its starting position.

        Note: This override is acceptable because population-based optimizers
        need to manage initialization across multiple sub-optimizers.

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
            self.p_current.pos_new = pos  # Property setter auto-appends

        # Check constraints
        if not self.conv.not_in_constraint(pos):
            max_tries = 100
            for _ in range(max_tries):
                pos = self.p_current.init.move_random_typed()
                if self.conv.not_in_constraint(pos):
                    break
            self.p_current.pos_current = pos
            self.p_current.pos_new = pos  # Property setter auto-appends
            if self.p_current.pos_new_list:
                self.p_current.pos_new_list[-1] = pos

        # Track position on main optimizer (property setter auto-appends)
        self.pos_new = pos

        return pos

    def evaluate_init(self, score_new: float) -> None:
        """Evaluate during initialization phase.

        Note: This override is acceptable because population-based optimizers
        need to track scores across multiple sub-optimizers.

        Tracks score on both main optimizer and current individual.
        """
        # Track on main optimizer (this increments nth_trial)
        self._track_score(score_new)

        # Track on current individual
        self.p_current.evaluate_init(score_new)

        # Update tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

    # =========================================================================
    # Template Method Implementation - NO iterate() override!
    # =========================================================================

    def _setup_iteration(self):
        """Set up current iteration by selecting individual and generating position.

        Called lazily by the first _iterate_*_batch() method.
        Decides between mutation and recombination, then generates the
        new position via the selected operation.
        """
        if self._iteration_setup_done:
            return

        self.n_ind = len(self.individuals)

        # Single individual: just mutate
        if self.n_ind == 1:
            self.p_current = self.individuals[0]
            self._current_new_pos = self.p_current.iterate()
            self._iteration_setup_done = True
            return

        # Sort and select random individual
        self.sort_pop_best_score()
        self.rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[self.rnd_int]

        # Decide: mutation or recombination
        total_rate = self.mutation_rate + self.crossover_rate
        rand = self._rng.uniform(0, total_rate)

        pos_count_before = len(self.p_current.pos_new_list)

        if rand <= self.mutation_rate:
            # Mutation: use individual's iterate (hill climbing with adaptive sigma)
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

                self.p_current.pos_new = pos_new  # Property setter auto-appends
        else:
            # Recombination
            pos_new = self._cross()

        self._current_new_pos = pos_new
        self._iteration_setup_done = True

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Generate continuous values by delegating to current sub-optimizer.

        Returns the continuous portion of the sub-optimizer's position.

        Returns
        -------
        np.ndarray
            New continuous values from mutation or recombination.
        """
        self._setup_iteration()
        return self._current_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
        """Generate categorical indices by delegating to current sub-optimizer.

        Returns the categorical portion of the sub-optimizer's position.

        Returns
        -------
        np.ndarray
            New category indices from mutation or recombination.
        """
        self._setup_iteration()
        return self._current_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self) -> np.ndarray:
        """Generate discrete indices by delegating to current sub-optimizer.

        Returns the discrete portion of the sub-optimizer's position.

        Returns
        -------
        np.ndarray
            New discrete indices from mutation or recombination.
        """
        self._setup_iteration()
        return self._current_new_pos[self._discrete_mask]

    def _evaluate(self, score_new: float) -> None:
        """Evaluate current individual.

        Delegates to the individual's evaluate method which handles
        personal best tracking and sigma adaptation.

        Also resets iteration state for the next iteration.

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

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._current_new_pos = None

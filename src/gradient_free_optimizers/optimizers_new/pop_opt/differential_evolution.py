# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Differential Evolution (DE) Optimizer.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
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


class DifferentialEvolutionOptimizer(BasePopulationOptimizer):
    """Differential Evolution optimization algorithm.

    Dimension Support:
        - Continuous: YES (differential mutation)
        - Categorical: YES (probabilistic parent selection)
        - Discrete: YES (differential mutation with rounding)

    Evolves a population using vector differences between randomly selected
    individuals. The mutation creates donor vectors that are combined with
    target vectors through crossover.

    The DE mutation equation is:
        mutant = x1 + F * (x2 - x3)

    For categorical dimensions, arithmetic operations on category indices
    are meaningless, so we use probabilistic selection from the three parents.

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
        initialize: dict[str, int] | None = None,
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
            population=population,
        )

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Create population of individuals
        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        # Initialize RNG for reproducibility
        self._rng = np.random.default_rng(self.random_seed)

    def mutation(self) -> np.ndarray:
        """Generate mutant vector using type-aware differential mutation.

        For continuous and discrete-numerical dimensions, uses standard DE
        mutation: x_1 + F * (x_2 - x_3).

        For categorical dimensions, uses probabilistic parent selection since
        arithmetic operations on category indices are meaningless.

        Returns
        -------
        np.ndarray
            Mutant vector.
        """
        # Need at least 3 individuals for DE mutation
        if len(self.individuals) < 3:
            # Fallback: use current individual's position with noise
            if self.p_current.pos_best is not None:
                return self.p_current.move_climb_typed(self.p_current.pos_best)
            return self.p_current.init.move_random_typed()

        # Select 3 distinct individuals
        ind_selected = random.sample(self.individuals, 3)
        x_1, x_2, x_3 = (ind.pos_best for ind in ind_selected)

        # Guard against None positions
        if x_1 is None or x_2 is None or x_3 is None:
            return self.p_current.init.move_random_typed()

        # Fast path for legacy mode (all discrete-numerical)
        if self.conv.is_legacy_mode:
            mutant = np.array(x_1) + self.mutation_rate * (
                np.array(x_2) - np.array(x_3)
            )
            return mutant

        # Type-aware mutation for mixed dimension types
        from gradient_free_optimizers._dimension_types import DimensionType

        mutant = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            if dim_type == DimensionType.CATEGORICAL:
                # Probabilistic selection from parents for categorical dims
                if random.random() < self.mutation_rate:
                    # Pick randomly from one of the three parents
                    mutant.append(random.choice([x_1[idx], x_2[idx], x_3[idx]]))
                else:
                    mutant.append(x_1[idx])
            else:
                # Standard DE mutation for continuous and discrete-numerical
                mutant.append(x_1[idx] + self.mutation_rate * (x_2[idx] - x_3[idx]))

        return np.array(mutant)

    def discrete_recombination(self, parent_pos_l, crossover_rates=None):
        """Combine parent positions using discrete recombination.

        For each dimension, randomly select from one of the parents.

        Parameters
        ----------
        parent_pos_l : list of np.ndarray
            List of parent positions (typically [target, mutant]).
        crossover_rates : list of float, optional
            Probability weights for each parent.

        Returns
        -------
        np.ndarray
            Trial vector.
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

    def conv2pos_typed(self, pos):
        """Convert position to valid position with proper types.

        Clips values to bounds and ensures correct data types for each dimension.

        Parameters
        ----------
        pos : np.ndarray
            Position to convert.

        Returns
        -------
        np.ndarray
            Valid position.
        """
        if self.conv.is_legacy_mode:
            n_zeros = [0] * len(self.conv.max_positions)
            return np.clip(pos, n_zeros, self.conv.max_positions).astype(int)

        from gradient_free_optimizers._dimension_types import DimensionType

        pos_new = []
        for idx, dim_type in enumerate(self.conv.dim_types):
            bounds = self.conv.dim_infos[idx].bounds
            val = pos[idx]

            if dim_type == DimensionType.CONTINUOUS:
                # Clip to bounds, keep as float
                pos_new.append(np.clip(val, bounds[0], bounds[1]))
            else:
                # Discrete or categorical: clip and convert to int
                pos_new.append(int(np.clip(round(val), bounds[0], bounds[1])))

        return np.array(pos_new)

    def _constraint_loop(self, position: np.ndarray) -> np.ndarray:
        """Ensure position satisfies constraints.

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
        """Generate trial vector via mutation and crossover.

        DE iteration:
        1. Select target vector (round-robin through population)
        2. Generate mutant vector using differential mutation
        3. Create trial vector via crossover between target and mutant
        4. Selection happens in evaluate() (greedy)

        Returns
        -------
        np.ndarray
            Trial vector for evaluation.
        """
        # Select target individual (round-robin)
        self.p_current = self.individuals[self.nth_trial % len(self.individuals)]
        target_vector = self.p_current.pos_current

        # Guard against None target
        if target_vector is None:
            pos_new = self.p_current.init.move_random_typed()
            self.p_current.pos_new = pos_new  # Property setter auto-appends
            self.pos_new = pos_new  # Property setter auto-appends
            return pos_new

        # Generate mutant vector
        mutant_vector = self.mutation()

        # Crossover: combine target and mutant
        crossover_rates = [1 - self.crossover_rate, self.crossover_rate]
        pos_new = self.discrete_recombination(
            [target_vector, mutant_vector],
            crossover_rates,
        )

        # Convert to valid position (clip and round as needed)
        pos_new = self.conv2pos_typed(pos_new)

        # Handle constraints
        pos_new = self._constraint_loop(pos_new)
        pos_new = self.conv2pos_typed(pos_new)

        # Track on individual (property setter auto-appends)
        self.p_current.pos_new = pos_new

        # Track on main optimizer (property setter auto-appends)
        self.pos_new = pos_new

        return pos_new

    def _evaluate(self, score_new: float) -> None:
        """Evaluate trial vector and perform selection.

        DE uses greedy selection: the trial vector replaces the target
        only if it has a better (or equal) score.

        Parameters
        ----------
        score_new : float
            Score of the trial vector.
        """
        # DE greedy selection is handled by individual's evaluate
        # which updates pos_current only if score improved
        self.p_current.evaluate(score_new)

        # Update global tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

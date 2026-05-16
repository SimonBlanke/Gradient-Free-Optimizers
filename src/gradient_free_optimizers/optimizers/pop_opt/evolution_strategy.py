# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Evolution Strategy (ES) Optimizer."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from gradient_free_optimizers._array_backend import array, ndarray
from gradient_free_optimizers._array_backend import random as arr_random

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
        boundary: str = "clip",
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
            boundary=boundary,
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
        self._rng = arr_random.default_rng(self.random_seed)

        # State for iterate
        self.rnd_int = 0
        self.n_ind = len(self.individuals)

        # Iteration state for template method coordination
        self._iteration_setup_done = False
        self._current_new_pos = None
        self._current_candidate_sigma = None
        self._offspring_records = []

        if self.offspring < 1:
            raise ValueError("offspring must be at least 1")
        if self.replace_parents and self.offspring < len(self.individuals):
            raise ValueError("replace_parents=True requires offspring >= population")

    def _discrete_recombination(self, parent_pos_l, crossover_rates=None):
        """Combine parent positions using discrete recombination.

        For each dimension, randomly select from one of the parents.

        Parameters
        ----------
        parent_pos_l : list of ndarray
            List of parent positions.
        crossover_rates : list of float, optional
            Probability weights for each parent. If None, uniform selection.

        Returns
        -------
        ndarray
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

        return array(result)

    def _restore_candidate_state(self, individual) -> None:
        """Reset per-candidate mutation state after generating an offspring."""
        if getattr(individual, "_original_epsilon", None) is not None:
            individual.epsilon = individual._original_epsilon
            individual._original_epsilon = None

        if hasattr(individual, "_iteration_setup_done"):
            individual._iteration_setup_done = False
        if hasattr(individual, "_use_random_restart"):
            individual._use_random_restart = False

    def _individual_sigma(self, individual):
        """Return an individual's strategy step size when available."""
        if hasattr(individual, "sigma"):
            return individual.sigma
        return getattr(individual, "epsilon", None)

    def _candidate_sigma(self, individual):
        """Return the candidate strategy step size when available."""
        if hasattr(individual, "sigma_new"):
            return individual.sigma_new
        return self._individual_sigma(individual)

    def _set_individual_sigma(self, individual, sigma) -> None:
        """Set strategy step size on individuals that expose it."""
        if sigma is None:
            return

        if hasattr(individual, "sigma"):
            individual.sigma = sigma
        if hasattr(individual, "sigma_new"):
            individual.sigma_new = sigma

    def _mutation_offspring(self):
        """Generate one offspring by mutating the current parent."""
        pos_new = self.p_current._iterate()
        sigma_new = self._candidate_sigma(self.p_current)
        self._restore_candidate_state(self.p_current)
        return pos_new, sigma_new

    def _cross(self):
        """Generate one offspring by recombining the current parent.

        Selects a second parent different from the current one and performs
        discrete recombination. Population replacement is deferred until the
        generation has collected ``offspring`` evaluated candidates.

        Returns
        -------
        tuple[ndarray, float]
            New position from recombination and the inherited sigma.
        """
        # Select a second parent different from current
        available = [
            i
            for i, individual in enumerate(self.pop_sorted)
            if i != self.rnd_int and individual._pos_current is not None
        ]

        if not available:
            # Fallback to mutation if recombination cannot be formed.
            return self._mutation_offspring()

        rnd_int2 = random.choice(available)

        p_sec = self.pop_sorted[rnd_int2]

        # Guard against None positions
        if self.p_current._pos_current is None or p_sec._pos_current is None:
            return self._mutation_offspring()

        # Recombine the two parents
        two_best_pos = [self.p_current._pos_current, p_sec._pos_current]
        pos_new = self._discrete_recombination(two_best_pos)

        # Handle constraints
        if not self.conv.not_in_constraint(pos_new):
            pos_new = self.p_current.move_climb_typed(pos_new)

        pos_new = self._clip_position(
            pos_new,
            reference_position=self.p_current._pos_current,
        )
        self.p_current._pos_new = pos_new
        return pos_new, self._individual_sigma(self.p_current)

    def _select_generation(self) -> None:
        """Apply (mu, lambda) or (mu + lambda) selection after a generation."""
        generation = self._offspring_records[: self.offspring]
        remaining = self._offspring_records[self.offspring :]

        parent_records = []
        for individual in self.individuals:
            if individual._pos_current is None:
                continue
            parent_records.append(
                {
                    "individual": individual,
                    "position": individual._pos_current.copy(),
                    "score": individual._score_current,
                    "sigma": self._individual_sigma(individual),
                    "is_parent": True,
                }
            )

        selection_pool = (
            generation if self.replace_parents else parent_records + generation
        )
        selection_pool.sort(
            key=lambda record: (
                record["score"] if record["score"] is not None else float("-inf")
            ),
            reverse=True,
        )

        selected = selection_pool[: len(self.individuals)]
        selected_parent_ids = {
            id(record["individual"])
            for record in selected
            if record.get("is_parent", False)
        }

        assigned_slot_ids = set()
        self._sort_pop_best_score()
        replacement_slots = list(reversed(self.pop_sorted))

        for record in selected:
            if record.get("is_parent", False):
                continue

            slot = self._replacement_slot(
                record,
                selected_parent_ids,
                assigned_slot_ids,
                replacement_slots,
            )
            self._apply_generation_record(slot, record)
            assigned_slot_ids.add(id(slot))

        self._offspring_records = remaining

    def _replacement_slot(
        self,
        record,
        selected_parent_ids,
        assigned_slot_ids,
        replacement_slots,
    ):
        """Choose which individual receives a selected offspring."""
        target = record["individual"]
        target_id = id(target)
        if target_id not in selected_parent_ids and target_id not in assigned_slot_ids:
            return target

        for individual in replacement_slots:
            individual_id = id(individual)
            if (
                individual_id not in selected_parent_ids
                and individual_id not in assigned_slot_ids
            ):
                return individual

        return target

    def _apply_generation_record(self, individual, record) -> None:
        """Copy a selected offspring into a population slot."""
        score = record["score"]
        position = record["position"].copy()

        individual._pos_current = position
        individual._score_current = score
        self._set_individual_sigma(individual, record["sigma"])

        if individual._pos_best is None or score > individual._score_best:
            individual._pos_best = position.copy()
            individual._score_best = score

    def _on_init_pos(self, position) -> None:
        """Initialize current individual with the given position.

        Args:
            position: The initialization position from CoreOptimizer.init_pos()
        """
        # Select individual via round-robin (use nth_init-1 since it was incremented)
        nth_pop = (self.nth_init - 1) % len(self.individuals)
        self.p_current = self.individuals[nth_pop]

        # Track position on current individual
        self.p_current._pos_new = position.copy()
        self.p_current._pos_current = position.copy()

    def _on_evaluate_init(self, score_new: float) -> None:
        """Evaluate during initialization phase.

        Delegates evaluation to the current individual for individual-level tracking.

        Args:
            score_new: Score of the most recently evaluated init position
        """
        # Track on current individual
        self.p_current._score_new = score_new

        # Update individual's best if this is better
        if self.p_current._pos_best is None or score_new > self.p_current._score_best:
            self.p_current._pos_best = self.p_current._pos_new.copy()
            self.p_current._score_best = score_new

        # Update individual's current
        self.p_current._score_current = score_new

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
            pos_new, sigma_new = self._mutation_offspring()
            self._current_new_pos = pos_new
            self._current_candidate_sigma = sigma_new
            self._iteration_setup_done = True
            return

        # Sort and select random parent
        self._sort_pop_best_score()
        self.rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[self.rnd_int]

        # Decide: mutation or recombination
        total_rate = self.mutation_rate + self.crossover_rate
        pos_count_before = len(self.p_current._pos_new_list)

        if total_rate <= 0:
            pos_new, sigma_new = self._mutation_offspring()
        elif self._rng.uniform(0, total_rate) <= self.mutation_rate:
            pos_new, sigma_new = self._mutation_offspring()

            if not self.conv.not_in_constraint(pos_new):
                while len(self.p_current._pos_new_list) > pos_count_before:
                    self.p_current._pos_new_list.pop()

                max_tries = 100
                for _ in range(max_tries):
                    pos_new = self.p_current.init.move_random_typed()
                    if self.conv.not_in_constraint(pos_new):
                        break

                self.p_current._pos_new = pos_new
                sigma_new = self._individual_sigma(self.p_current)
        else:
            pos_new, sigma_new = self._cross()

        self._current_new_pos = pos_new
        self._current_candidate_sigma = sigma_new
        self._iteration_setup_done = True

    def _iterate_continuous_batch(self) -> ndarray:
        """Generate continuous values by delegating to current sub-optimizer.

        Returns the continuous portion of the sub-optimizer's position.

        Returns
        -------
        ndarray
            New continuous values from mutation or recombination.
        """
        self._setup_iteration()
        return self._current_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> ndarray:
        """Generate categorical indices by delegating to current sub-optimizer.

        Returns the categorical portion of the sub-optimizer's position.

        Returns
        -------
        ndarray
            New category indices from mutation or recombination.
        """
        self._setup_iteration()
        return self._current_new_pos[self._categorical_mask]

    def _iterate_discrete_batch(self) -> ndarray:
        """Generate discrete indices by delegating to current sub-optimizer.

        Returns the discrete portion of the sub-optimizer's position.

        Returns
        -------
        ndarray
            New discrete indices from mutation or recombination.
        """
        self._setup_iteration()
        return self._current_new_pos[self._discrete_mask]

    def _on_evaluate(self, score_new: float) -> None:
        """Evaluate current individual.

        Delegates to the individual's evaluate method which handles
        personal best tracking and sigma adaptation.

        Also resets iteration state for the next iteration.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # The candidate position was appended when it was generated. Set the
        # backing field directly so score tracking aligns with this evaluation
        # without adding a duplicate position entry.
        self.p_current.__dict__["_CoreOptimizer__pos_new"] = self._pos_new
        self.p_current._track_score(score_new)
        self.p_current._update_best(self._pos_new, score_new)

        self._offspring_records.append(
            {
                "individual": self.p_current,
                "position": self._pos_new.copy(),
                "score": score_new,
                "sigma": self._current_candidate_sigma,
                "is_parent": False,
            }
        )

        if len(self._offspring_records) >= self.offspring:
            self._select_generation()

        # Update global tracking
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._current_new_pos = None
        self._current_candidate_sigma = None

    def _iterate_batch(self, n):
        """Generate n positions via independent mutation/recombination."""
        positions = []
        self._batch_individual_refs = []
        self._batch_candidate_sigmas = []
        for _ in range(n):
            self._iteration_setup_done = False
            self._setup_iteration()
            positions.append(self._clip_position(self._current_new_pos))
            self._batch_individual_refs.append(self.p_current)
            self._batch_candidate_sigmas.append(self._current_candidate_sigma)
            self._iteration_setup_done = False
            self._current_new_pos = None
            self._current_candidate_sigma = None
        return positions

    def _evaluate_batch(self, positions, scores):
        """Process batch results, restoring the correct individual for each."""
        for i, (pos, score) in enumerate(zip(positions, scores)):
            self.p_current = self._batch_individual_refs[i]
            self._current_candidate_sigma = self._batch_candidate_sigmas[i]
            self._pos_new = pos
            self._evaluate(score)
        self._batch_candidate_sigmas = []

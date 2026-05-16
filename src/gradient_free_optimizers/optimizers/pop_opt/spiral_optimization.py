# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Spiral Optimization Algorithm."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from gradient_free_optimizers._array_backend import array, ndarray, zeros
from gradient_free_optimizers._dimension_types import DimensionType

from ._spiral import Spiral, rotation
from .base_population_optimizer import BasePopulationOptimizer

if TYPE_CHECKING:
    pass


class SpiralOptimization(BasePopulationOptimizer):
    """Spiral Optimization Algorithm.

    Dimension Support:
        - Continuous: YES (spiral movement)
        - Categorical: YES (decay factor as switch probability)
        - Discrete: YES (spiral movement, rounded)

    Particles move in a spiral pattern toward the current best position.
    The spiral movement combines rotation and contraction to balance
    exploration and exploitation.

    The spiral equation is:
        new_pos = denormalize(
            center_norm + radius * decay * rotation(current_norm - center_norm)
        )

    Where:
        - center is the global best position
        - radius is a dimensionless normalized search radius
        - decay contracts over time (decay_rate^iteration)
        - rotation_degrees controls the angular stride of the spiral trajectory

    Template Method Pattern:
        This optimizer follows the Template Method Pattern by implementing
        _iterate_*_batch() methods instead of overriding iterate().
        The spiral position is computed once per iteration, then portions
        are extracted for each dimension type.

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
        Number of particles in the swarm.
    decay_rate : float, default=0.99
        Rate at which spiral radius contracts per iteration.
        Values closer to 1 cause slower contraction (more exploration).
    spiral_radius : float, default=1.0
        Initial radius multiplier in normalized search-space coordinates.
    rotation_degrees : float, default=90.0
        Rotation angle applied to the normalized offset at each step.
    """

    name = "Spiral Optimization"
    _name_ = "spiral_optimization"
    __name__ = "SpiralOptimization"

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
        decay_rate: float = 0.99,
        spiral_radius: float = 1.0,
        rotation_degrees: float = 90.0,
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

        self.decay_rate = decay_rate
        self.spiral_radius = spiral_radius
        self.rotation_degrees = rotation_degrees

        # Create population of spiral particles
        self.particles = self._create_population(Spiral)
        self.optimizers = self.particles

        # Set decay_rate on all particles upfront
        for p in self.particles:
            p.decay_rate = self.decay_rate
            p.spiral_radius = self.spiral_radius
            p.rotation_degrees = self.rotation_degrees
            p.decay_factor = self.spiral_radius

        # Center position (best found so far)
        self.center_pos = None
        self.center_score = None

        # Iteration state for template method coordination
        self._iteration_setup_done = False
        self._spiral_new_pos = None
        self._decay_factor = self.spiral_radius
        self._spiral_decay_before_candidate = None
        self._spiral_decay_before_constraint_retries = None

    def _on_init_pos(self, position) -> None:
        """Initialize current particle with the given position.

        Sets up particle decay rate and assigns the position.

        Args:
            position: The initialization position from CoreOptimizer.init_pos()
        """
        # Select particle via round-robin (use nth_init-1 since it was incremented)
        nth_pop = (self.nth_init - 1) % len(self.particles)
        self.p_current = self.particles[nth_pop]
        self.p_current.decay_rate = self.decay_rate
        self.p_current.spiral_radius = self.spiral_radius
        self.p_current.rotation_degrees = self.rotation_degrees

        # Track position on current particle
        self.p_current._pos_new = position.copy()
        self.p_current._pos_current = position.copy()

    def _on_evaluate_init(self, score_new: float) -> None:
        """Evaluate during initialization phase.

        Delegates evaluation to the current particle for particle-level tracking.

        Args:
            score_new: Score of the most recently evaluated init position
        """
        # Track on current particle
        self.p_current._score_new = score_new

        # Update particle's best if this is better
        if self.p_current._pos_best is None or score_new > self.p_current._score_best:
            self.p_current._pos_best = self.p_current._pos_new.copy()
            self.p_current._score_best = score_new

        # Update particle's current
        self.p_current._score_current = score_new

    def _sort_pop_personal_best_score(self) -> None:
        """Sort particles by their personal-best score, best first."""
        indexed = list(enumerate(self.particles))
        indexed.sort(
            key=lambda x: (
                x[1]._score_best if x[1]._pos_best is not None else float("-inf")
            ),
            reverse=True,
        )
        self.pop_sorted = [self.particles[i] for i, _ in indexed]

    def _update_center_from_personal_best(self, force: bool = False) -> None:
        """Update the spiral center from the best particle personal best."""
        self._sort_pop_personal_best_score()
        best_particle = self.pop_sorted[0]

        if best_particle._pos_best is None or best_particle._score_best is None:
            return

        if (
            force
            or self.center_score is None
            or best_particle._score_best > self.center_score
        ):
            self.center_pos = best_particle._pos_best.copy()
            self.center_score = best_particle._score_best

    def _on_finish_initialization(self) -> None:
        """Set up initial center position from best particle.

        Called by CoreOptimizer.finish_initialization() after all init
        positions have been evaluated. Initializes the spiral center
        to the best position found during initialization.

        Note: DO NOT set search_state here - CoreOptimizer handles that.
        """
        self._update_center_from_personal_best(force=True)

    def _setup_iteration(self) -> None:
        """Set up current iteration by selecting particle and computing spiral position.

        Called lazily by the first _iterate_*_batch() method.
        Computes the full spiral position once, which is then extracted
        by the individual batch methods.
        """
        if self._iteration_setup_done:
            return

        # Select current particle (round-robin)
        self.p_current = self.particles[self.nth_trial % len(self.particles)]

        # Update global best reference for this particle
        self._update_center_from_personal_best()
        self.p_current.global_pos_best = self.center_pos

        # Compute full spiral position
        self._spiral_decay_before_candidate = self._decay_factor
        self._spiral_new_pos = self._compute_spiral_position()

        self._iteration_setup_done = True

    def _compute_spiral_position(self) -> ndarray:
        """Compute new position using spiral movement equation.

        The spiral equation is:
            new_pos = denormalize(
                center_norm + radius * decay * rotation(current_norm - center_norm)
            )

        Returns
        -------
        np.ndarray
            New position after spiral movement.
        """
        # Random restart check
        if random.random() < self.rand_rest_p:
            return self.p_current.init.move_random_typed()

        # Guard against None positions during early iterations
        if self.center_pos is None or self.p_current._pos_current is None:
            return self.p_current.init.move_random_typed()

        # Update decay factor
        self._decay_factor *= self.decay_rate

        center = array(self.center_pos)
        current = array(self.p_current._pos_current)
        center_norm = self._normalize_position(center)
        current_norm = self._normalize_position(current)

        rot = rotation(
            len(center_norm),
            current_norm - center_norm,
            rotation_degrees=self.rotation_degrees,
        )
        new_norm = center_norm + self._decay_factor * rot

        return self._denormalize_position(new_norm)

    def _compute_dimension_bounds(self) -> tuple[ndarray, ndarray]:
        """Compute lower and upper internal bounds for each dimension.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper bounds in the optimizer's internal position space.
        """
        n_dims = len(self.search_space)
        lower = zeros(n_dims)
        upper = zeros(n_dims)

        for i, info in enumerate(self.conv.dim_infos):
            if info.dim_type.is_continuous_like:
                lower[i] = info.bounds[0]
                upper[i] = info.bounds[1]
            elif info.dim_type == DimensionType.CATEGORICAL:
                lower[i] = 0
                upper[i] = info.size - 1
            else:
                lower[i] = 0
                upper[i] = info.size - 1

        return lower, upper

    def _normalize_position(self, position) -> ndarray:
        """Map an internal position to normalized [0, 1] coordinates."""
        lower, upper = self._compute_dimension_bounds()
        normalized = []

        for value, lo, hi in zip(position, lower, upper):
            span = hi - lo
            if span > 0:
                normalized.append((value - lo) / span)
            else:
                normalized.append(0.0)

        return array(normalized)

    def _denormalize_position(self, normalized_position) -> ndarray:
        """Map normalized [0, 1] coordinates back to internal positions."""
        lower, upper = self._compute_dimension_bounds()
        position = []

        for value, lo, hi in zip(normalized_position, lower, upper):
            span = hi - lo
            if span > 0:
                position.append(lo + value * span)
            else:
                position.append(lo)

        return array(position)

    def _clear_spiral_candidate_cache(self) -> None:
        self._iteration_setup_done = False
        self._spiral_new_pos = None

    def _on_constraint_retry(self, rejected_position) -> None:
        if self._spiral_decay_before_constraint_retries is None:
            self._spiral_decay_before_constraint_retries = (
                self._spiral_decay_before_candidate
            )

        self._clear_spiral_candidate_cache()

    def _on_constraint_retry_fallback(self) -> None:
        if self._spiral_decay_before_constraint_retries is not None:
            self._decay_factor = self._spiral_decay_before_constraint_retries

        self._clear_spiral_candidate_cache()

    def _on_constraint_retry_success(self, accepted_position) -> None:
        self._spiral_decay_before_constraint_retries = None
        self._spiral_decay_before_candidate = None

    def _iterate_continuous_batch(self) -> ndarray:
        """Generate continuous values using spiral movement.

        Returns the continuous portion of the spiral-computed position.

        Returns
        -------
        np.ndarray
            New continuous values from spiral movement.
        """
        self._setup_iteration()
        return self._spiral_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> ndarray:
        """Generate categorical indices using spiral decay as switch probability.

        For categorical dimensions, the decay factor magnitude determines the
        probability of switching to a different category. Lower decay (more
        exploitation) means lower switch probability.

        Returns
        -------
        np.ndarray
            New category indices.
        """
        self._setup_iteration()

        # Get current categorical indices
        current = self.p_current._pos_current[self._categorical_mask]

        new_cats = []
        for i, cur_idx in enumerate(current):
            # Get max valid index for this categorical dimension
            max_idx = self._categorical_sizes[i] - 1

            # Decay factor determines switch probability
            # Higher decay_factor (early iterations) = more exploration
            # Lower decay_factor (later iterations) = more exploitation
            switch_prob = min(1.0, self._decay_factor / 10.0)

            if random.random() < switch_prob:
                # Switch to a random category
                new_cats.append(random.randint(0, max_idx))
            else:
                # Keep current category
                new_cats.append(int(cur_idx))

        return array(new_cats)

    def _iterate_discrete_batch(self) -> ndarray:
        """Generate discrete indices using spiral movement.

        Returns the discrete portion of the spiral-computed position.

        Returns
        -------
        np.ndarray
            New discrete indices from spiral movement.
        """
        self._setup_iteration()
        return self._spiral_new_pos[self._discrete_mask]

    def _on_evaluate(self, score_new: float) -> None:
        """Evaluate current particle and update center if improved.

        Updates the spiral center to the best position found so far.
        Also resets iteration state for the next iteration.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Track position on particle
        self.p_current._pos_new = self._pos_new

        # Delegate to current particle's evaluate
        self.p_current._evaluate(score_new)

        if self.search_state == "iter":
            self._update_center_from_personal_best()

        # Update global tracking
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._spiral_new_pos = None
        self._spiral_decay_before_candidate = None
        self._spiral_decay_before_constraint_retries = None

    def _iterate_batch(self, n):
        """Generate n positions by cycling through spiral particles.

        The decay factor is applied once per batch rather than once per
        position. Otherwise a batch of size n would decay the spiral
        n times, contracting it much faster than in serial mode.
        """
        self._update_center_from_personal_best()
        positions = []
        self._batch_particle_indices = []

        saved_decay = self._decay_factor
        for i in range(n):
            idx = (self.nth_trial + i) % len(self.particles)
            self._batch_particle_indices.append(idx)
            self.p_current = self.particles[idx]
            self.p_current.global_pos_best = self.center_pos
            # Use the same decay factor for all positions in this batch
            self._decay_factor = saved_decay
            pos = self._compute_spiral_position()
            positions.append(self._clip_position(pos))

        # Apply decay exactly once for the entire batch
        self._decay_factor = saved_decay * self.decay_rate

        return positions

    def _evaluate_batch(self, positions, scores):
        """Process batch results, restoring the correct particle for each."""
        for i, (pos, score) in enumerate(zip(positions, scores)):
            self.p_current = self.particles[self._batch_particle_indices[i]]
            self._pos_new = pos
            self._evaluate(score)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Spiral Optimization Algorithm.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL

Template Method Pattern Compliance:
    - Does NOT override iterate() - uses CoreOptimizer's orchestration
    - Implements _iterate_*_batch() for dimension-type-aware spiral movement
    - Overrides init_pos()/evaluate_init() for population management (acceptable)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

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
        new_pos = center + decay * rotation(current - center)

    Where:
        - center is the global best position
        - decay contracts over time (decay_rate^iteration)
        - rotation creates the spiral trajectory

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
        population: int = 10,
        decay_rate: float = 0.99,
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

        self.decay_rate = decay_rate

        # Create population of spiral particles
        self.particles = self._create_population(Spiral)
        self.optimizers = self.particles

        # Set decay_rate on all particles upfront
        for p in self.particles:
            p.decay_rate = self.decay_rate

        # Center position (best found so far)
        self.center_pos = None
        self.center_score = None

        # Iteration state for template method coordination
        self._iteration_setup_done = False
        self._spiral_new_pos = None
        self._decay_factor = 3.0  # Initial scaling factor

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

    def _on_finish_initialization(self) -> None:
        """Set up initial center position from best particle.

        Called by CoreOptimizer.finish_initialization() after all init
        positions have been evaluated. Initializes the spiral center
        to the best position found during initialization.

        Note: DO NOT set search_state here - CoreOptimizer handles that.
        """
        self._sort_pop_best_score()
        self.center_pos = self.pop_sorted[0]._pos_current
        self.center_score = self.pop_sorted[0]._score_current

    # =========================================================================
    # Template Method Implementation - NO iterate() override!
    # =========================================================================

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
        self._sort_pop_best_score()
        self.p_current.global_pos_best = self.pop_sorted[0]._pos_current

        # Compute full spiral position
        self._spiral_new_pos = self._compute_spiral_position()

        self._iteration_setup_done = True

    def _compute_spiral_position(self) -> np.ndarray:
        """Compute new position using spiral movement equation.

        The spiral equation is:
            new_pos = center + decay * rotation(current - center)

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

        # Compute step rate based on search space size
        scales = self._compute_dimension_scales()
        step_rate = self._decay_factor * scales / 1000

        # Compute rotated offset from center
        center = np.array(self.center_pos)
        current = np.array(self.p_current._pos_current)
        rot = rotation(len(center), current - center)

        # Combine rotation with decay
        offset = step_rate * rot
        new_pos = center + offset

        return new_pos

    def _compute_dimension_scales(self) -> np.ndarray:
        """Compute scale factors for each dimension.

        Returns
        -------
        np.ndarray
            Scale factor for each dimension based on its type and bounds.
        """
        n_dims = len(self.search_space)
        scales = np.zeros(n_dims)
        dim_names = list(self.search_space.keys())

        for i, name in enumerate(dim_names):
            dim_def = self.search_space[name]

            if isinstance(dim_def, tuple) and len(dim_def) == 2:
                # Continuous: use range
                scales[i] = dim_def[1] - dim_def[0]
            elif isinstance(dim_def, list):
                # Categorical: use number of categories
                scales[i] = len(dim_def) - 1
            elif isinstance(dim_def, np.ndarray):
                # Discrete: use max index
                scales[i] = len(dim_def) - 1

        return scales

    def _iterate_continuous_batch(self) -> np.ndarray:
        """Generate continuous values using spiral movement.

        Returns the continuous portion of the spiral-computed position.

        Returns
        -------
        np.ndarray
            New continuous values from spiral movement.
        """
        self._setup_iteration()
        return self._spiral_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> np.ndarray:
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

        return np.array(new_cats)

    def _iterate_discrete_batch(self) -> np.ndarray:
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

        # Update center if better position found
        if self.search_state == "iter":
            self._sort_pop_best_score()
            if self.pop_sorted[0]._score_current is not None:
                if (
                    self.center_score is None
                    or self.pop_sorted[0]._score_current > self.center_score
                ):
                    self.center_pos = self.pop_sorted[0]._pos_current
                    self.center_score = self.pop_sorted[0]._score_current

        # Delegate to current particle's evaluate
        self.p_current._evaluate(score_new)

        # Update global tracking
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._spiral_new_pos = None

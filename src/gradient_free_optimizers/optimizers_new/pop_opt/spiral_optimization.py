# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Spiral Optimization Algorithm.

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ._spiral import Spiral

if TYPE_CHECKING:
    pass


class SpiralOptimization(BasePopulationOptimizer):
    """Spiral Optimization Algorithm.

    Dimension Support:
        - Continuous: YES (spiral movement)
        - Categorical: YES (with appropriate handling)
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
        # (needed for particles that might be used in iterate() without init_pos())
        for p in self.particles:
            p.decay_rate = self.decay_rate

        # Center position (best found so far)
        self.center_pos = None
        self.center_score = None

    def init_pos(self) -> np.ndarray:
        """Initialize current particle and return its starting position.

        Sets up particle decay rate and returns initial position.

        Returns
        -------
        np.ndarray
            Initial position for the current particle.
        """
        nth_pop = self.nth_trial % len(self.particles)
        self.p_current = self.particles[nth_pop]
        self.p_current.decay_rate = self.decay_rate

        # Get initial position from particle (this tracks on particle)
        if self.p_current.nth_init < len(self.p_current.init.init_positions_l):
            pos = self.p_current.init_pos()
        else:
            # Fall back to random position when particle has no more init positions
            pos = self.p_current.init.move_random_typed()
            self.p_current.pos_current = pos
            self.p_current.pos_new = pos
            self.p_current.pos_new_list.append(pos)

        # Check constraints - if violated, find valid position and replace
        if not self.conv.not_in_constraint(pos):
            max_tries = 100
            for _ in range(max_tries):
                pos = self.p_current.init.move_random_typed()
                if self.conv.not_in_constraint(pos):
                    break
            # Replace the invalid position in particle's tracking with valid one
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

        Tracks score on both main optimizer and current particle.
        """
        # Track on main optimizer (this increments nth_trial)
        self._track_score(score_new)

        # Track on current particle
        self.p_current.evaluate_init(score_new)

        # Update tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

    def finish_initialization(self) -> None:
        """Transition from initialization to iteration phase.

        Sets up the initial center position based on the best
        particle found during initialization.
        """
        self.sort_pop_best_score()
        self.center_pos = self.pop_sorted[0].pos_current
        self.center_score = self.pop_sorted[0].score_current

        self.search_state = "iter"

    def iterate(self) -> np.ndarray:
        """Move current particle in spiral pattern toward center.

        Selects the next particle in round-robin order and computes
        the new position using spiral movement equations.

        Returns
        -------
        np.ndarray
            New position for the current particle.
        """
        # Select current particle (round-robin)
        self.p_current = self.particles[self.nth_trial % len(self.particles)]

        # Update global best reference
        self.sort_pop_best_score()
        self.p_current.global_pos_best = self.pop_sorted[0].pos_current

        # Remember position count before move (to fix tracking if needed)
        pos_count_before = len(self.p_current.pos_new_list)

        # Compute new position using spiral movement
        pos_new = self.p_current.move_spiral(self.center_pos)

        # Check constraints - if valid, we're done
        if self.conv.not_in_constraint(pos_new):
            self.pos_new = pos_new
            self.pos_new_list.append(pos_new)
            return pos_new

        # Constraint violated - restore position count and try fallback
        while len(self.p_current.pos_new_list) > pos_count_before:
            self.p_current.pos_new_list.pop()

        pos_new = self.p_current.iterate()

        # Check constraint on fallback
        if self.conv.not_in_constraint(pos_new):
            self.pos_new = pos_new
            self.pos_new_list.append(pos_new)
            return pos_new

        # Still violated - try random positions as last resort
        max_tries = 100
        while len(self.p_current.pos_new_list) > pos_count_before:
            self.p_current.pos_new_list.pop()

        for _ in range(max_tries):
            pos_new = self.p_current.init.move_random_typed()
            if self.conv.not_in_constraint(pos_new):
                break

        # Track final position (even if still violated - let Search handle it)
        self.p_current.pos_new = pos_new
        self.p_current.pos_new_list.append(pos_new)
        self.pos_new = pos_new
        self.pos_new_list.append(pos_new)
        return pos_new

    def _evaluate(self, score_new: float) -> None:
        """Evaluate current particle and update center if improved.

        Updates the spiral center to the best position found so far.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Update center if better position found
        if self.search_state == "iter":
            if self.pop_sorted[0].score_current is not None:
                if self.center_score is None or self.pop_sorted[0].score_current > self.center_score:
                    self.center_pos = self.pop_sorted[0].pos_current
                    self.center_score = self.pop_sorted[0].score_current

        # Delegate to current particle's evaluate
        self.p_current.evaluate(score_new)

        # Update global tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Particle Swarm Optimization (PSO).

Supports: CONTINUOUS, CATEGORICAL, DISCRETE_NUMERICAL
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ._particle import Particle
from .base_population_optimizer import BasePopulationOptimizer

if TYPE_CHECKING:
    pass


class ParticleSwarmOptimizer(BasePopulationOptimizer):
    """Particle Swarm Optimization algorithm.

    Dimension Support:
        - Continuous: YES (velocity-based movement)
        - Categorical: YES (velocity interpreted as switch probability)
        - Discrete: YES (velocity-based with rounding)

    Simulates a swarm of particles moving through the search space. Each
    particle is influenced by its own best position (cognitive component)
    and the global best position found by the swarm (social component).

    The velocity update equation is:
        v = inertia * v
          + cognitive_weight * r1 * (personal_best - position)
          + social_weight * r2 * (global_best - position)

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
    inertia : float, default=0.5
        Weight for maintaining current velocity direction (momentum).
    cognitive_weight : float, default=0.5
        Weight for attraction toward personal best position.
    social_weight : float, default=0.5
        Weight for attraction toward global best position.
    temp_weight : float, default=0.2
        Temperature weight for exploration randomness.
    """

    name = "Particle Swarm Optimization"
    _name_ = "particle_swarm_optimization"
    __name__ = "ParticleSwarmOptimizer"

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
        inertia: float = 0.5,
        cognitive_weight: float = 0.5,
        social_weight: float = 0.5,
        temp_weight: float = 0.2,
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

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight

        # Create population of particles
        self.particles = self._create_population(Particle)
        self.optimizers = self.particles

        # Set particle parameters upfront on all particles
        # (needed for particles that might be used in iterate() without init_pos())
        n_dims = len(self.conv.search_space)
        for p in self.particles:
            p.inertia = self.inertia
            p.cognitive_weight = self.cognitive_weight
            p.social_weight = self.social_weight
            p.temp_weight = self.temp_weight
            p.rand_rest_p = self.rand_rest_p
            p.velo = np.zeros(n_dims)

    def init_pos(self) -> np.ndarray:
        """Initialize current particle and return its starting position.

        Sets up particle parameters (inertia, weights) and initializes
        velocity to zero for smooth startup.

        Returns
        -------
        np.ndarray
            Initial position for the current particle.
        """
        nth_pop = self.nth_trial % len(self.particles)
        self.p_current = self.particles[nth_pop]

        # Set particle parameters
        self.p_current.inertia = self.inertia
        self.p_current.cognitive_weight = self.cognitive_weight
        self.p_current.social_weight = self.social_weight
        self.p_current.temp_weight = self.temp_weight
        self.p_current.rand_rest_p = self.rand_rest_p

        # Initialize velocity to zero
        n_dims = self.conv.n_dimensions
        self.p_current.velo = np.zeros(n_dims)

        # Get initial position from particle (this tracks on particle)
        if self.p_current.nth_init < len(self.p_current.init.init_positions_l):
            pos = self.p_current.init_pos()
        else:
            # Fall back to random position when particle has no more init positions
            pos = self.p_current.init.move_random_typed()
            self.p_current.pos_current = pos
            self.p_current.pos_new = pos  # Property setter auto-appends

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

        # Track position on main optimizer (property setter auto-appends)
        self.pos_new = pos

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

    def iterate(self) -> np.ndarray:
        """Move current particle based on velocity update equations.

        Selects the next particle in round-robin order, updates the
        global best reference, and computes the new position using
        PSO velocity equations.

        Returns
        -------
        np.ndarray
            New position for the current particle.
        """
        # Select current particle (round-robin)
        self.p_current = self.particles[self.nth_trial % len(self.particles)]

        # Update global best reference for this particle
        self.sort_pop_best_score()
        self.p_current.global_pos_best = self.pop_sorted[0].pos_best

        # Remember position count before move (to fix tracking if needed)
        pos_count_before = len(self.p_current.pos_new_list)

        # Compute new position using PSO velocity update
        pos_new = self.p_current.move_linear()

        # Check constraints - if valid, we're done
        if self.conv.not_in_constraint(pos_new):
            self.pos_new = pos_new  # Property setter auto-appends
            return pos_new

        # Constraint violated - restore position count and try fallback
        while len(self.p_current.pos_new_list) > pos_count_before:
            self.p_current.pos_new_list.pop()

        pos_new = self.p_current.move_climb_typed(pos_new)

        # Check constraint on fallback
        if self.conv.not_in_constraint(pos_new):
            self.pos_new = pos_new  # Property setter auto-appends
            return pos_new

        # Still violated - try random positions as last resort
        max_tries = 100
        while len(self.p_current.pos_new_list) > pos_count_before:
            self.p_current.pos_new_list.pop()

        for _ in range(max_tries):
            pos_new = self.p_current.init.move_random_typed()
            if self.conv.not_in_constraint(pos_new):
                break

        # Track final position (property setters auto-append)
        self.p_current.pos_new = pos_new
        self.pos_new = pos_new
        return pos_new

    def _evaluate(self, score_new: float) -> None:
        """Evaluate current particle and update its personal/global best.

        Delegates to the particle's evaluate method which handles
        personal best tracking.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Delegate to current particle's evaluate
        self.p_current.evaluate(score_new)

        # Update global tracking
        self._update_best(self.pos_new, score_new)
        self._update_current(self.pos_new, score_new)

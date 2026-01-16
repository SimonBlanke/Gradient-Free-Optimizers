# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gradient_free_optimizers._array_backend import zeros
from gradient_free_optimizers._init_utils import get_default_initialize

from ._particle import Particle
from .base_population_optimizer import BasePopulationOptimizer


class ParticleSwarmOptimizer(BasePopulationOptimizer):
    """Particle Swarm Optimization algorithm.

    Simulates a swarm of particles moving through the search space. Each
    particle is influenced by its own best position and the global best
    position found by the swarm.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial positions.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
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
        Weight for maintaining current velocity direction.
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
        if initialize is None:
            initialize = get_default_initialize()

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )

        self.population = population
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight

        self.particles = self._create_population(Particle)
        self.optimizers = self.particles

    @BasePopulationOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.particles)

        self.p_current = self.particles[nth_pop]

        self.p_current.inertia = self.inertia
        self.p_current.cognitive_weight = self.cognitive_weight
        self.p_current.social_weight = self.social_weight
        self.p_current.temp_weight = self.temp_weight
        self.p_current.rand_rest_p = self.rand_rest_p

        self.p_current.velo = zeros(len(self.conv.max_positions))

        return self.p_current.init_pos()

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        """Move current particle based on velocity update equations."""
        while True:
            self.p_current = self.particles[self.nth_trial % len(self.particles)]

            self.sort_pop_best_score()
            self.p_current.global_pos_best = self.pop_sorted[0].pos_best

            pos_new = self.p_current.move_linear()

            if self.conv.not_in_constraint(pos_new):
                return pos_new
            pos_new = self.p_current.move_climb_typed(pos_new)

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)

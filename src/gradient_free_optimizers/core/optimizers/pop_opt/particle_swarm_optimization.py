# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ._particle import Particle


class ParticleSwarmOptimizer(BasePopulationOptimizer):
    name = "Particle Swarm Optimization"
    _name_ = "particle_swarm_optimization"
    __name__ = "ParticleSwarmOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        population=10,
        inertia=0.5,
        cognitive_weight=0.5,
        social_weight=0.5,
        temp_weight=0.2,
    ):
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

        self.p_current.velo = np.zeros(len(self.conv.max_positions))

        return self.p_current.init_pos()

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        while True:
            self.p_current = self.particles[
                self.nth_trial % len(self.particles)
            ]

            self.sort_pop_best_score()
            self.p_current.global_pos_best = self.pop_sorted[0].pos_best

            pos_new = self.p_current.move_linear()

            if self.conv.not_in_constraint(pos_new):
                return pos_new
            pos_new = self.p_current.move_climb(pos_new)

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)

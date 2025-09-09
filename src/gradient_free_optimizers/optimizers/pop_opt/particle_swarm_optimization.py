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
        # v2 detection and placeholder for legacy core init
        from ..._search_space.base import BaseSearchSpace
        from ..._search_space.v2_integration import V2CompatConverter, V2Initializer

        is_v2 = isinstance(search_space, BaseSearchSpace)
        ss_arg = search_space if not is_v2 else {"__v2_placeholder__": np.array([0])}

        super().__init__(
            search_space=ss_arg,
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

        # v2 override of converter/initializer
        self._v2_active = is_v2
        if self._v2_active:
            names, dims = search_space._build_dimensions()
            self.conv = V2CompatConverter(names=names, dims=dims, constraints=constraints)
            self.init = V2Initializer(self.conv, initialize)
            self._v2_space = search_space

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

    # Override to pass v2 search spaces to particles
    def _create_population(self, Optimizer):
        if not getattr(self, "_v2_active", False):
            return super()._create_population(Optimizer)

        if isinstance(self.population, int):
            pop_size = self.population
        else:
            pop_size = len(self.population)
        diff_init = pop_size - self.init.n_inits
        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

        if isinstance(self.population, int):
            from .base_population_optimizer import split

            distributed_init_positions = split(self.init.init_positions_l, self.population)

            population = []
            for init_positions in distributed_init_positions:
                init_values = self.conv.positions2values(init_positions)
                init_paras = self.conv.values2paras(init_values)
                population.append(
                    Optimizer(
                        self._v2_space,
                        rand_rest_p=self.rand_rest_p,
                        initialize={"warm_start": init_paras},
                        constraints=self.constraints,
                    )
                )
        else:
            population = self.population

        return population

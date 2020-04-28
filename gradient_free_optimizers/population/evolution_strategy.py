# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from math import floor, ceil
import numpy as np

from . import ParticleSwarmOptimizer
from ..local import HillClimbingPositioner


class EvolutionStrategyOptimizer(ParticleSwarmOptimizer):
    def __init__(self, init_positions, space_dim, opt_para):
        super().__init__(init_positions, space_dim, opt_para)
        self.n_positioners = self._opt_args_.individuals

        self.n_mutations = floor(self.n_positioners * self._opt_args_.mutation_rate)
        self.n_crossovers = ceil(self.n_positioners * self._opt_args_.crossover_rate)

        self.n_iter_rank_indiv = 3

    def _mutate(self):
        return self.p_current.move_climb(self.p_current.pos_current)

    def _random_cross(self, array_list):
        n_arrays = len(array_list)
        size = array_list[0].size
        shape = array_list[0].shape

        choice = np.random.randint(n_arrays, size=size).reshape(shape).astype(bool)
        return np.choose(choice, array_list)

    def _cross(self):
        p_rest = [self.p_rest[i].pos_current for i in range(len(self.p_rest))]
        pos_current_list = [self.p_current.pos_current] + p_rest
        pos = self._random_cross(pos_current_list)
        self.p_current.pos_new = pos

        return pos

    def _choose_evo(self):
        total_rate = self._opt_args_.mutation_rate + self._opt_args_.crossover_rate
        rand = np.random.uniform(low=0, high=total_rate)

        if len(self.init_positions) == 1 or rand <= self._opt_args_.mutation_rate:
            self._sort_()
            self._choose_next_pos()

            return self._mutate()
        else:
            self._sort_best()
            self._choose_next_pos()

            return self._cross()

    def iterate(self, nth_iter):
        self._base_iterate(nth_iter)
        pos = self._choose_evo()

        return pos

    def evaluate(self, score_new):
        self.p_current.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()


class Individual(HillClimbingPositioner):
    def __init__(self, space_dim, _opt_args_):
        super().__init__(space_dim, _opt_args_)

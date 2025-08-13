# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from collections import OrderedDict

from ...local_opt import HillClimbingOptimizer


def sort_list_idx(list_):
    list_np = np.array(list_)
    idx_sorted = list(list_np.argsort()[::-1])
    return idx_sorted


class PowellsMethod(HillClimbingOptimizer):
    name = "Powell's Method"
    _name_ = "powells_method"
    __name__ = "PowellsMethod"

    optimizer_type = "global"
    computationally_expensive = False

    def __init__(
            self,
            search_space,
            initialize={"grid": 4, "random": 2, "vertices": 4},
            constraints=[],
            random_state=None,
            rand_rest_p=0,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
            iters_p_dim=10,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
        )

        self.iters_p_dim = iters_p_dim
        self.current_search_dim = -1
        self.search_directions = []
        self.initial_position = None
        self.last_best_position = None
        self.cycle_count = 0

    def finish_initialization(self):
        self.nth_iter_ = -1
        self.nth_iter_current_dim = 0
        self.search_state = "iter"
        self.search_directions = []
        for i in range(self.conv.n_dimensions):
            v = np.zeros(self.conv.n_dimensions)
            v[i] = 1.0
            self.search_directions.append(v)

        if self.positions_valid:
            idx = np.argmax(self.scores_valid)
            self.initial_position = self.positions_valid[idx]
            self.last_best_position = self.initial_position.copy()

    def new_dim(self):
        self.current_search_dim += 1
        if self.current_search_dim >= len(self.search_directions):
            self.current_search_dim = 0
            self.cycle_count += 1
            if self.cycle_count > 0 and self.positions_valid:
                self.update_search_directions()

        idx_sort = sort_list_idx(self.scores_valid)
        self.powells_pos = self.positions_valid[idx_sort[0]]
        self.powells_scores = self.scores_valid[idx_sort[0]]
        self.nth_iter_current_dim = 0
        self.setup_line_search()

    def update_search_directions(self):
        if self.initial_position is None or not self.positions_valid:
            return
        idx = np.argmax(self.scores_valid)
        current = self.positions_valid[idx]
        disp = np.array(current) - np.array(self.last_best_position)
        if np.any(disp):
            norm = np.linalg.norm(disp)
            if norm > 0:
                new_dir = disp / norm
                self.search_directions = self.search_directions[1:] + [new_dir]
            self.last_best_position = current.copy()

    def setup_line_search(self):
        dir_vec = self.search_directions[self.current_search_dim]
        sp1d = OrderedDict()
        for i, name in enumerate(self.conv.para_names):
            if abs(dir_vec[i]) > 1e-10:
                sp1d[name] = np.array(self.conv.search_space_positions[i])
            else:
                sp1d[name] = np.array([self.powells_pos[i]])

        self.hill_climb = HillClimbingOptimizer(
            search_space=sp1d,
            initialize={"random": 0},
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
        )

        init_idx = self.hill_climb.conv.value2position(self.powells_pos)
        self.hill_climb.positions_valid = [init_idx]
        self.hill_climb.scores_valid = [self.powells_scores]
        self.hill_climb.best_pos = init_idx
        self.hill_climb.best_score = self.powells_scores
        self.hill_climb.current_pos = init_idx
        self.hill_climb.pos_current = init_idx

    @HillClimbingOptimizer.track_new_pos
    @HillClimbingOptimizer.random_iteration
    def iterate(self):
        self.nth_iter_ += 1
        self.nth_iter_current_dim += 1
        if self.nth_iter_ % self.iters_p_dim == 0:
            self.new_dim()

        next_idx = self.hill_climb.iterate()
        if next_idx is None:
            next_idx = self.hill_climb.init_pos()

        pos_new = np.array(self.hill_climb.conv.position2value(next_idx))
        if self.conv.not_in_constraint(pos_new):
            return pos_new
        return self.move_climb(pos_new, epsilon=self.epsilon, distribution=self.distribution)

    @HillClimbingOptimizer.track_new_score
    def evaluate(self, score_new):
        if self.current_search_dim == -1:
            super(HillClimbingOptimizer, self).evaluate(score_new)
        else:
            self.hill_climb.evaluate(score_new)
            super(HillClimbingOptimizer, self).evaluate(score_new)

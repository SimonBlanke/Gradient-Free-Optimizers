# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from gradient_free_optimizers._init_utils import get_default_initialize

from ..core_optimizer.converter import ArrayLike
from .hill_climbing_optimizer import HillClimbingOptimizer


def _arrays_equal(a: Any, b: Any) -> bool:
    """Check if two arrays are element-wise equal."""
    if hasattr(a, "__len__") and hasattr(b, "__len__"):
        if len(a) != len(b):
            return False
        return all(x == y for x, y in zip(a, b))
    return a == b


def sort_list_idx(list_: list[float]) -> list[int]:
    """Return indices that would sort the list in descending order."""
    indexed = list(enumerate(list_))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in indexed]


def centroid(array_list: list[ArrayLike]) -> list[float]:
    """Calculate centroid of a list of arrays."""
    n_dims = len(array_list[0])
    result = []

    for idx in range(n_dims):
        center_dim_pos = [arr[idx] for arr in array_list]
        center_dim_mean = sum(center_dim_pos) / len(center_dim_pos)
        result.append(center_dim_mean)

    return result


class DownhillSimplexOptimizer(HillClimbingOptimizer):
    """Nelder-Mead downhill simplex optimizer.

    Maintains a simplex of n+1 points in n-dimensional space and iteratively
    transforms it through reflection, expansion, contraction, and shrinkage
    operations to find the optimum.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default=None
        Strategy for generating initial simplex vertices.
        If None, uses {"grid": 4, "random": 2, "vertices": 4}.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    alpha : float, default=1
        Reflection coefficient.
    gamma : float, default=2
        Expansion coefficient.
    beta : float, default=0.5
        Contraction coefficient.
    sigma : float, default=0.5
        Shrinkage coefficient.
    """

    name = "Downhill Simplex"
    _name_ = "downhill_simplex"
    __name__ = "DownhillSimplexOptimizer"

    optimizer_type = "local"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        alpha: float = 1,
        gamma: float = 2,
        beta: float = 0.5,
        sigma: float = 0.5,
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

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma

        self.n_simp_positions = len(self.conv.search_space) + 1
        self.simp_positions = []

        self.simplex_step = 0

        diff_init = self.n_simp_positions - self.init.n_inits
        if diff_init > 0:
            self.init.add_n_random_init_pos(diff_init)

    def finish_initialization(self) -> None:
        idx_sorted = sort_list_idx(self.scores_valid)
        self.simplex_pos = [self.positions_valid[idx] for idx in idx_sorted]
        self.simplex_scores = [self.scores_valid[idx] for idx in idx_sorted]

        self.simplex_step = 1

        self.i_x_0 = 0
        self.i_x_N_1 = -2
        self.i_x_N = -1

        self.search_state = "iter"

    @HillClimbingOptimizer.track_new_pos
    def iterate(self) -> ArrayLike:
        """Generate next simplex position via reflection/expansion/contraction.

        Uses type-aware position conversion that handles discrete, continuous,
        and categorical dimensions appropriately.
        """
        simplex_stale = all(
            _arrays_equal(self.simplex_pos[0], array) for array in self.simplex_pos
        )

        if simplex_stale:
            idx_sorted = sort_list_idx(self.scores_valid)
            self.simplex_pos = [self.positions_valid[idx] for idx in idx_sorted]
            self.simplex_scores = [self.scores_valid[idx] for idx in idx_sorted]

            self.simplex_step = 1

        if self.simplex_step == 1:
            idx_sorted = sort_list_idx(self.simplex_scores)
            self.simplex_pos = [self.simplex_pos[idx] for idx in idx_sorted]
            self.simplex_scores = [self.simplex_scores[idx] for idx in idx_sorted]

            self.center_array = centroid(self.simplex_pos[:-1])

            r_pos = self.center_array + self.alpha * (
                self.center_array - self.simplex_pos[-1]
            )
            self.r_pos = self.conv2pos_typed(r_pos)
            pos_new = self.r_pos

        elif self.simplex_step == 2:
            e_pos = self.center_array + self.gamma * (
                self.center_array - self.simplex_pos[-1]
            )
            self.e_pos = self.conv2pos_typed(e_pos)
            self.simplex_step = 1

            pos_new = self.e_pos

        elif self.simplex_step == 3:
            # iter Contraction
            c_pos = self.h_pos + self.beta * (self.center_array - self.h_pos)
            c_pos = self.conv2pos_typed(c_pos)

            pos_new = c_pos

        elif self.simplex_step == 4:
            # iter Shrink
            pos = self.simplex_pos[self.compress_idx]
            pos = pos + self.sigma * (self.simplex_pos[0] - pos)

            pos_new = self.conv2pos_typed(pos)

        if self.conv.not_in_constraint(pos_new):
            return pos_new

        return self.move_climb_typed(
            pos_new, epsilon=self.epsilon, distribution=self.distribution
        )

    @HillClimbingOptimizer.track_new_score
    def evaluate(self, score_new: float) -> None:
        """Evaluate score and update simplex state machine."""
        if self.simplex_step != 0:
            self.prev_pos = self.positions_valid[-1]

        if self.simplex_step == 1:
            # self.r_pos = self.prev_pos
            self.r_score = score_new

            if self.r_score > self.simplex_scores[0]:
                self.simplex_step = 2

            elif self.r_score > self.simplex_scores[-2]:
                # if r is better than x N-1
                self.simplex_pos[-1] = self.r_pos
                self.simplex_scores[-1] = self.r_score
                self.simplex_step = 1

            if self.simplex_scores[-1] > self.r_score:
                self.h_pos = self.simplex_pos[-1]
                self.h_score = self.simplex_scores[-1]
            else:
                self.h_pos = self.r_pos
                self.h_score = self.r_score

            self.simplex_step = 3

        elif self.simplex_step == 2:
            self.e_score = score_new

            if self.e_score > self.r_score:
                self.simplex_scores[-1] = self.e_pos
            elif self.r_score > self.e_score:
                self.simplex_scores[-1] = self.r_pos
            else:
                self.simplex_scores[-1] = random.choice([self.e_pos, self.r_pos])[0]

        elif self.simplex_step == 3:
            # eval Contraction
            self.c_pos = self.prev_pos
            self.c_score = score_new

            if self.c_score > self.simplex_scores[-1]:
                self.simplex_scores[-1] = self.c_score
                self.simplex_pos[-1] = self.c_pos

                self.simplex_step = 1

            else:
                # start Shrink
                self.simplex_step = 4
                self.compress_idx = 0

        elif self.simplex_step == 4:
            # eval Shrink
            self.simplex_scores[self.compress_idx] = score_new
            self.simplex_pos[self.compress_idx] = self.prev_pos

            self.compress_idx += 1

            if self.compress_idx == self.n_simp_positions:
                self.simplex_step = 1

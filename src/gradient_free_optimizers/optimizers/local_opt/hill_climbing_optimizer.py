# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..base_optimizer import BaseOptimizer
from ..._search_space.base import BaseSearchSpace
from ..._search_space.v2_integration import V2CompatConverter, V2Initializer
from ..._search_space.dimensions import CategoricalDimension, FixedDimension


def max_list_idx(list_):
    max_item = max(list_)
    max_item_idx = [i for i, j in enumerate(list_) if j == max_item]
    return max_item_idx[-1:][0]


class HillClimbingOptimizer(BaseOptimizer):
    name = "Hill Climbing"
    _name_ = "hill_climbing"
    __name__ = "HillClimbingOptimizer"

    optimizer_type = "local"
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
    ):
        # If v2 dataclass space is provided, pass a dummy placeholder to base init
        ss_arg = search_space
        is_v2 = isinstance(search_space, BaseSearchSpace)
        if is_v2:
            ss_arg = {"__v2_placeholder__": np.array([0])}

        super().__init__(
            search_space=ss_arg,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
        )
        self.epsilon = epsilon
        self.distribution = distribution
        self.n_neighbours = n_neighbours

        # v2 integration: if search_space is dataclass-based, swap converter/initializer
        self._v2_active = is_v2
        if self._v2_active:
            names, dims = search_space._build_dimensions()
            self.conv = V2CompatConverter(names=names, dims=dims, constraints=constraints)
            self.init = V2Initializer(self.conv, initialize)
            # Seed RNG for deterministic steps if desired
            self._rng = np.random.default_rng(self.random_seed)

    @BaseOptimizer.track_new_pos
    @BaseOptimizer.random_iteration
    def iterate(self):
        return self.move_climb(
            self.pos_current,
            epsilon=self.epsilon,
            distribution=self.distribution,
        )

    @BaseOptimizer.track_new_score
    def evaluate(self, score_new):
        BaseOptimizer.evaluate(self, score_new)
        if len(self.scores_valid) == 0:
            return

        modZero = self.nth_trial % self.n_neighbours == 0
        if modZero:
            score_new_list_temp = self.scores_valid[-self.n_neighbours :]
            pos_new_list_temp = self.positions_valid[-self.n_neighbours :]

            idx = max_list_idx(score_new_list_temp)
            score = score_new_list_temp[idx]
            pos = pos_new_list_temp[idx]

            self._eval2current(pos, score)
            self._eval2best(pos, score)

    # ---------- v2 overrides ----------
    def conv2pos(self, pos):
        if not getattr(self, "_v2_active", False):
            return super().conv2pos(pos)
        # z-space: clamp to [0, 1]
        return np.clip(np.asarray(pos, dtype=float), 0.0, 1.0)

    def move_random(self):
        if not getattr(self, "_v2_active", False):
            return super().move_random()
        # sample z by sampling native values and mapping to z to avoid bias
        z = self.conv._conv.sample_z(self._rng)
        return np.array(z, dtype=float)

    def move_climb(self, pos, epsilon=0.03, distribution="normal", epsilon_mod=1):
        if not getattr(self, "_v2_active", False):
            return super().move_climb(pos, epsilon, distribution, epsilon_mod)

        # z-space arithmetic for continuous-like dims; neighbor flips for categoricals
        z = np.array(pos, dtype=float)
        values = self.conv.position2value(z)

        while True:
            z_new = z.copy()
            for i, dim in enumerate(self.conv.dims):  # type: ignore[attr-defined]
                kind = getattr(dim, "kind", "")
                if kind in ("real", "integer", "distribution"):
                    sigma = max(1e-12, float(epsilon) * float(epsilon_mod))
                    if distribution == "normal":
                        step = self._rng.normal(0.0, sigma)
                    elif distribution == "laplace":
                        step = self._rng.laplace(0.0, sigma)
                    elif distribution == "logistic":
                        step = self._rng.logistic(0.0, sigma)
                    elif distribution == "gumbel":
                        step = self._rng.gumbel(0.0, sigma)
                    else:
                        step = self._rng.normal(0.0, sigma)
                    z_new[i] = float(np.clip(z_new[i] + step, 0.0, 1.0))
                elif kind == "categorical":
                    v_new = dim.perturb(values[i], scale=float(epsilon) * float(epsilon_mod), rng=self._rng)
                    z_new[i] = float(dim.value_to_z(v_new))
                elif kind == "fixed":
                    z_new[i] = float(dim.value_to_z(values[i]))
                else:
                    # default safe clamp
                    z_new[i] = float(np.clip(z_new[i], 0.0, 1.0))

            if self.conv.not_in_constraint(z_new):
                return z_new
            epsilon_mod *= 1.01

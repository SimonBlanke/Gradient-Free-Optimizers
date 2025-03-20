# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..smb_opt.smbo import SMBO

from scipy.spatial.distance import cdist


class LipschitzFunction:
    def __init__(self, position_l):
        self.position_l = position_l

    def find_best_slope(self, X_sample, Y_sample):
        slopes = []

        len_sample = len(X_sample)
        for i in range(len_sample):
            for j in range(i + 1, len_sample):
                x_sample1, y_sample1 = X_sample[i], Y_sample[i]
                x_sample2, y_sample2 = X_sample[j], Y_sample[j]

                if y_sample1 != y_sample2 and np.prod((x_sample1 - x_sample2)) != 0:
                    slopes.append(
                        abs(y_sample1 - y_sample2) / abs(x_sample1 - x_sample2)
                    )

        if not slopes:
            return 1
        return np.max(slopes)

    def calculate(self, X_sample, Y_sample, score_best):
        lip_c = self.find_best_slope(X_sample, Y_sample)

        positions_np = np.array(self.position_l)
        samples_np = np.array(X_sample)

        pos_dist = cdist(positions_np, samples_np) * lip_c

        upper_bound_l = pos_dist
        upper_bound_l += np.array(Y_sample)

        mx = np.ma.masked_array(upper_bound_l, mask=upper_bound_l == 0)
        upper_bound_l = mx.min(1).reshape(1, -1).T
        upper_bound_l[upper_bound_l <= score_best] = -np.inf

        return upper_bound_l


class LipschitzOptimizer(SMBO):
    name = "Lipschitz Optimizer"
    _name_ = "lipschitz_optimizer"
    __name__ = "LipschitzOptimizer"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        replacement=True,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

    def finish_initialization(self):
        self.all_pos_comb = self._all_possible_pos()
        return super().finish_initialization()

    @SMBO.track_new_pos
    @SMBO.track_X_sample
    def iterate(self):
        self.pos_comb = self._sampling(self.all_pos_comb)

        lip_func = LipschitzFunction(self.pos_comb)
        upper_bound_l = lip_func.calculate(
            self.X_sample, self.Y_sample, self.score_best
        )

        index_best = list(upper_bound_l.argsort()[::-1])
        all_pos_comb_sorted = self.pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

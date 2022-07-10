# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..smb_opt.smbo import SMBO
from ...search import Search

from scipy.spatial.distance import cdist


class LipschitzFunction:
    def __init__(self, position_l):
        self.position_l = position_l
        self.lip_c = None

    def lipschitz_constant(self):
        pass

    def calculate_1(self, X_sample, Y_sample):
        upper_bound_l = []

        for position_ in self.position_l:
            upper_bound = []

            for sample_x, sample_y in zip(X_sample, Y_sample):
                dist_ = cdist(position_.reshape(1, -1), sample_x.reshape(1, -1))

                upper_bound.append(sample_y + dist_ * self.lip_c)

            _upper_bound_ = np.array(upper_bound).sum()
            upper_bound_l.append(_upper_bound_)

        return np.array(upper_bound_l)

    def find_best_slope(self, X_sample, Y_sample):
        # slope = -np.inf

        slopes = [
            abs(y_sample1 - y_sample2) / abs(x_sample1 - x_sample2)
            for x_sample1, y_sample1 in zip(X_sample, Y_sample)
            for x_sample2, y_sample2 in zip(X_sample, Y_sample)
            if y_sample1 is not y_sample2
            if np.prod((x_sample1 - x_sample2)) != 0
        ]
        # print("\n\n -------- slopes \n", slopes, "\n\n")

        return np.max(slopes)

    def calculate(self, X_sample, Y_sample, score_best):
        self.lip_c = self.find_best_slope(X_sample, Y_sample)
        # print("\n self.lip_c ", self.lip_c, "\n")
        upper_bound_l = []

        M_A = np.array(self.position_l)
        M_B = np.array(X_sample)

        # print("\n X_sample \n", M_B.shape, "\n ")

        dist_ = cdist(M_A, M_B)

        dist_ = dist_ * self.lip_c
        # print("\n\n\n dist_ \n", dist_, dist_.shape, "\n")

        upper_bound_l = dist_
        # print("\n upper_bound_l \n", upper_bound_l, upper_bound_l.shape, "\n")
        # print("\n\n upper_bound_l 1 \n", upper_bound_l, upper_bound_l.shape, "\n")

        upper_bound_l += np.array(Y_sample)

        """
        n_samples = len(Y_sample)
        for _ in range(n_samples - 1):
            upper_bound_l[
                np.arange(len(upper_bound_l)), np.argmax(upper_bound_l, axis=1)
            ] = 0
        """
        # print("score_best", score_best)
        # print("\n upper_bound_l 1 \n", upper_bound_l, upper_bound_l.shape, "\n")

        mx = np.ma.masked_array(upper_bound_l, mask=upper_bound_l == 0)
        upper_bound_l = mx.min(1).reshape(1, -1).T
        # print("\n upper_bound_l 2 \n", upper_bound_l, upper_bound_l.shape, "\n")

        upper_bound_l[upper_bound_l <= score_best] = -np.inf

        # print("\n upper_bound_l 3 \n", upper_bound_l, upper_bound_l.shape, "\n")

        """
        upper_bound_l = np.sum(upper_bound_l, axis=1)
        print("\n upper_bound_l 3 \n", upper_bound_l, upper_bound_l.shape, "\n\n")
        """

        """
        for position_ in self.position_l:
            upper_bound = []

            for sample_x, sample_y in zip(X_sample, Y_sample):
                dist_ = cdist(position_.reshape(1, -1), sample_x.reshape(1, -1))

                upper_bound.append(sample_y + dist_ * self.lip_c)

            _upper_bound_ = np.array(upper_bound).sum()
            upper_bound_l.append(_upper_bound_)
        """

        # print("\n upper_bound_l \n", upper_bound_l, upper_bound_l.shape, "\n")
        # print("\n position_l \n", M_A, "\n ")

        return upper_bound_l


class LipschitzOptimizer(SMBO, Search):
    name = "Lipschitz Optimizer"
    _name_ = "lipschitz_optimizer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @SMBO.track_nth_iter
    @SMBO.track_X_sample
    def iterate(self):
        all_pos_comb = self._all_possible_pos()
        self.pos_comb = self._sampling(all_pos_comb)

        lip_func = LipschitzFunction(self.pos_comb)
        upper_bound_l = lip_func.calculate(
            self.X_sample, self.Y_sample, self.score_best
        )

        """
        print(
            "\n upper_bound_l \n", upper_bound_l, "\n shape", upper_bound_l.shape, "\n"
        )
        print("\n self.pos_comb \n", self.pos_comb, "\n")
        """
        index_best = list(upper_bound_l.argsort()[::-1])
        all_pos_comb_sorted = self.pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        # print("New pos", pos_best)
        return pos_best

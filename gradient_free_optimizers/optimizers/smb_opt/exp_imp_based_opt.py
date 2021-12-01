# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from scipy.stats import norm

from .smbo import SMBO


def normalize(array):
    num = array - array.min()
    den = array.max() - array.min()

    if den == 0:
        return np.random.random_sample(array.shape)
    else:
        return ((num / den) + 0) / 1


class ExpectedImprovementBasedOptimization(SMBO):
    def __init__(
        self,
        *args,
        xi=0.01,
        warm_start_smbo=None,
        sampling={"random": 1000000},
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.new_positions = []
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.sampling = sampling
        self.warnings = warnings

    def _expected_improvement(self):
        all_pos_comb = self._all_possible_pos()
        self.pos_comb = self._sampling(all_pos_comb)

        mu, sigma = self.regr.predict(self.pos_comb, return_std=True)
        # mu_sample = self.regr.predict(self.X_sample)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        Y_sample = normalize(np.array(self.Y_sample)).reshape(-1, 1)
        imp = mu - np.max(Y_sample) - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma != 0)

        exploit = imp * norm.cdf(Z)
        explore = sigma * norm.pdf(Z)

        exp_imp = exploit + explore
        exp_imp[sigma == 0.0] = 0.0

        return exp_imp[:, 0]

    def _propose_location(self):
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        if len(Y_sample) == 0:
            return self.move_random()

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

        exp_imp = self._expected_improvement()

        index_best = list(exp_imp.argsort()[::-1])
        all_pos_comb_sorted = self.pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    @SMBO.track_nth_iter
    @SMBO.track_X_sample
    def iterate(self):
        return self._propose_location()

    @SMBO.track_y_sample
    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

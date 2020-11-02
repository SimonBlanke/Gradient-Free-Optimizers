# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist


from .sbom import SBOM


class ExpectedImprovementBasedOptimization(SBOM):
    def __init__(
        self, search_space, xi=0.01, warm_start_sbom=None, rand_rest_p=0.03,
    ):
        super().__init__(search_space)
        self.new_positions = []
        self.xi = xi
        self.warm_start_sbom = warm_start_sbom
        self.rand_rest_p = rand_rest_p

    def _expected_improvement(self):
        # print("self.all_pos_comb", self.all_pos_comb.shape)

        mu, sigma = self.regr.predict(self.all_pos_comb, return_std=True)
        mu_sample = self.regr.predict(self.X_sample)

        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        mu_sample = mu_sample.reshape(-1, 1)

        mu_sample_opt = np.max(mu_sample)
        imp = mu - mu_sample_opt - self.xi

        Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma != 0)
        exp_imp = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        exp_imp[sigma == 0.0] = 0.0

        return exp_imp

    def _propose_location(self):
        """
        print(
            "\n self.X_sample \n", self.X_sample, np.array(self.X_sample).shape
        )
        print(
            "\n self.Y_sample \n", self.Y_sample, np.array(self.Y_sample).shape
        )
        """
        self.regr.fit(np.array(self.X_sample), np.array(self.Y_sample))

        exp_imp = self._expected_improvement()
        exp_imp = exp_imp[:, 0]
        # print("\n exp_imp \n", exp_imp)

        index_best = list(exp_imp.argsort()[::-1])
        all_pos_comb_sorted = self.all_pos_comb[index_best]
        # print("\n all_pos_comb_sorted \n", all_pos_comb_sorted)

        pos_best = all_pos_comb_sorted[0]

        # print("pos_best", pos_best)

        return pos_best

    @SBOM.track_nth_iter
    @SBOM.track_X_sample
    def iterate(self):
        return self._propose_location()

    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

        self.Y_sample.append(score_new)

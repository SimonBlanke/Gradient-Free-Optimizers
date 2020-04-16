# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist


from .sbom import SBOM


class BayesianOptimizer(SBOM):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.regr = self._opt_args_.gpr
        self.new_positions = []

    def expected_improvement(self):
        all_pos_comb_sampled = self.get_random_sample()

        mu, sigma = self.regr.predict(all_pos_comb_sampled, return_std=True)
        mu_sample = self.regr.predict(self.X_sample)

        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        mu_sample = mu_sample.reshape(-1, 1)

        mu_sample_opt = np.max(mu_sample)
        imp = mu - mu_sample_opt - self._opt_args_.xi

        Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma != 0)
        exp_imp = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        exp_imp[sigma == 0.0] = 0.0

        return exp_imp

    def propose_location(self, i, _cand_):
        self.regr.fit(self.X_sample, self.Y_sample)

        exp_imp = self.expected_improvement()
        exp_imp = exp_imp[:, 0]

        index_best = list(exp_imp.argsort()[::-1])
        all_pos_comb_sorted = self.all_pos_comb[index_best]

        pos_best = [all_pos_comb_sorted[0]]

        while len(pos_best) < self._opt_args_.skip_retrain(i):
            if all_pos_comb_sorted.shape[0] == 0:
                break

            dists = cdist(all_pos_comb_sorted, [pos_best[-1]], metric="cityblock")
            dists_norm = dists / dists.max()
            bool = np.squeeze(dists_norm > 0.25)
            all_pos_comb_sorted = all_pos_comb_sorted[bool]

            if len(all_pos_comb_sorted) > 0:
                pos_best.append(all_pos_comb_sorted[0])

        return pos_best

    def _iterate(self, i, _cand_):
        if i < self._opt_args_.start_up_evals:
            self.p_list[0].move_random(_cand_)
            self._optimizer_eval(_cand_, self.p_list[0])
            self._update_pos(_cand_, self.p_list[0])
        else:
            if len(self.new_positions) == 0:
                self.new_positions = self.propose_location(i, _cand_)

            self.p_list[0].pos_new = self.new_positions[0]
            self._optimizer_eval(_cand_, self.p_list[0])
            self._update_pos(_cand_, self.p_list[0])

            self.new_positions.pop(0)

        self.X_sample = np.vstack((self.X_sample, self.p_list[0].pos_new))
        self.Y_sample = np.vstack((self.Y_sample, self.p_list[0].score_new))

        return _cand_

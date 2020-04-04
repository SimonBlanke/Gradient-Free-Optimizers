# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from sklearn.neighbors import KernelDensity

from .sbom import SBOM


class TreeStructuredParzenEstimators(SBOM):
    def __init__(self, n_iter, opt_para):
        super().__init__(n_iter, opt_para)
        self.n_positioners = 1
        self.kd_best = KernelDensity()
        self.kd_worst = KernelDensity()

    def _get_samples(self):
        n_samples = self.X_sample.shape[0]

        n_best = int(n_samples * self._opt_args_.gamma_tpe)

        Y_sample = self.Y_sample[:, 0]
        index_best = Y_sample.argsort()[-n_best:][::-1]

        best_samples = self.X_sample[index_best]
        worst_samples = self.X_sample[~index_best]

        return best_samples, worst_samples

    def expected_improvement(self):
        logprob_best = self.kd_best.score_samples(self.all_pos_comb)
        logprob_worst = self.kd_worst.score_samples(self.all_pos_comb)

        prob_best = np.exp(logprob_best)
        prob_worst = np.exp(logprob_worst)

        return np.divide(
            prob_best, prob_worst, out=np.zeros_like(prob_worst), where=prob_worst != 0
        )

    def propose_location(self, cand):
        best_samples, worst_samples = self._get_samples()

        self.kd_best.fit(best_samples)
        self.kd_worst.fit(worst_samples)

        exp_imp = self.expected_improvement()
        index_best = list(exp_imp.argsort()[::-1])

        all_pos_comb_sorted = self.all_pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    def _iterate(self, i, _cand_):
        if i < self._opt_args_.start_up_evals:
            self.p_list[0].move_random(_cand_)
            self._optimizer_eval(_cand_, self.p_list[0])
            self._update_pos(_cand_, self.p_list[0])

        else:

            self.p_list[0].pos_new = self.propose_location(_cand_)
            self.p_list[0].score_new = _cand_.eval_pos(self.p_list[0].pos_new)

            self._optimizer_eval(_cand_, self.p_list[0])
            self._update_pos(_cand_, self.p_list[0])

        self.X_sample = np.vstack((self.X_sample, self.p_list[0].pos_new))
        self.Y_sample = np.vstack((self.Y_sample, self.p_list[0].score_new))

        return _cand_

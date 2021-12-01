# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from sklearn.neighbors import KernelDensity
from .smbo import SMBO


class TreeStructuredParzenEstimators(SMBO):
    name = "Tree Structured Parzen Estimators"

    def __init__(
        self,
        *args,
        gamma_tpe=0.2,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gamma_tpe = gamma_tpe
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.warnings = warnings

        kde_para = {
            "kernel": "gaussian",
            "bandwidth": 1,
            "rtol": 0.001,
        }

        self.kd_best = KernelDensity(**kde_para)
        self.kd_worst = KernelDensity(**kde_para)

        self.init_warm_start_smbo()

    def _get_samples(self):
        n_samples = len(self.X_sample)

        n_best = int(n_samples * self.gamma_tpe)

        Y_sample = np.array(self.Y_sample)
        index_best = Y_sample.argsort()[-n_best:]
        n_worst = int(n_samples - n_best)
        index_worst = Y_sample.argsort()[:n_worst]

        best_samples = [self.X_sample[i] for i in index_best]
        worst_samples = [self.X_sample[i] for i in index_worst]

        return best_samples, worst_samples

    def expected_improvement(self):
        all_pos_comb = self._all_possible_pos()
        self.pos_comb = self._sampling(all_pos_comb)

        logprob_best = self.kd_best.score_samples(self.pos_comb)
        logprob_worst = self.kd_worst.score_samples(self.pos_comb)

        prob_best = np.exp(logprob_best)
        prob_worst = np.exp(logprob_worst)

        WorstOverbest = np.divide(
            prob_worst,
            prob_best,
            out=np.zeros_like(prob_worst),
            where=prob_worst != 0,
        )

        exp_imp_inv = self.gamma_tpe + WorstOverbest * (1 - self.gamma_tpe)
        exp_imp = 1 / exp_imp_inv

        return exp_imp

    def propose_location(self):
        best_samples, worst_samples = self._get_samples()

        try:
            self.kd_best.fit(best_samples)
            self.kd_worst.fit(worst_samples)
        except ValueError:
            print("Error: Surrogate model cannot fit to samples")

        exp_imp = self.expected_improvement()
        index_best = list(exp_imp.argsort()[::-1])

        all_pos_comb_sorted = self.pos_comb[index_best]
        pos_best = all_pos_comb_sorted[0]

        return pos_best

    @SMBO.track_nth_iter
    @SMBO.track_X_sample
    @SMBO.random_restart
    def iterate(self):
        return self.propose_location()

    @SMBO.track_y_sample
    def evaluate(self, score_new):
        self.score_new = score_new

        self._evaluate_new2current(score_new)
        self._evaluate_current2best()

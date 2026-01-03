# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from gradient_free_optimizers._array_backend import array, exp, zeros_like

from .smbo import SMBO

# Use sklearn's KDE if available, otherwise native implementation
try:
    from sklearn.neighbors import KernelDensity

    SKLEARN_AVAILABLE = True
except ImportError:
    from gradient_free_optimizers._estimators import KernelDensityEstimator as KernelDensity

    SKLEARN_AVAILABLE = False


class TreeStructuredParzenEstimators(SMBO):
    name = "Tree Structured Parzen Estimators"
    _name_ = "tree_structured_parzen_estimators"
    __name__ = "TreeStructuredParzenEstimators"

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
        gamma_tpe=0.2,
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

        self.gamma_tpe = gamma_tpe

        # Initialize KDE - sklearn and native have different interfaces
        if SKLEARN_AVAILABLE:
            kde_params = {"kernel": "gaussian", "bandwidth": 1.0}
        else:
            kde_params = {"bandwidth": 1.0}

        self.kd_best = KernelDensity(**kde_params)
        self.kd_worst = KernelDensity(**kde_params)

    def finish_initialization(self):
        self.all_pos_comb = self._all_possible_pos()
        return super().finish_initialization()

    def _get_samples(self):
        n_samples = len(self.X_sample)
        n_best = max(round(n_samples * self.gamma_tpe), 1)

        Y_sample = array(self.Y_sample)
        index_best = Y_sample.argsort()[-n_best:]
        n_worst = int(n_samples - n_best)
        index_worst = Y_sample.argsort()[:n_worst]

        best_samples = [self.X_sample[i] for i in index_best]
        worst_samples = [self.X_sample[i] for i in index_worst]

        return best_samples, worst_samples

    def _expected_improvement(self):
        self.pos_comb = self._sampling(self.all_pos_comb)

        logprob_best = self.kd_best.score_samples(self.pos_comb)
        logprob_worst = self.kd_worst.score_samples(self.pos_comb)

        prob_best = exp(array(logprob_best))
        prob_worst = exp(array(logprob_worst))

        # Match original np.divide(prob_worst, prob_best, out=zeros, where=prob_worst != 0)
        # Only divide where prob_worst != 0; output 0 otherwise
        WorstOverbest = zeros_like(prob_worst)
        for i in range(len(prob_worst)):
            if prob_worst[i] != 0:
                if prob_best[i] != 0:
                    WorstOverbest[i] = prob_worst[i] / prob_best[i]
                else:
                    # prob_best == 0 but prob_worst != 0 -> inf (like numpy)
                    WorstOverbest[i] = float('inf')

        exp_imp_inv = self.gamma_tpe + WorstOverbest * (1 - self.gamma_tpe)
        exp_imp = 1 / exp_imp_inv

        return exp_imp

    def _training(self):
        best_samples, worst_samples = self._get_samples()

        self.kd_best.fit(best_samples)
        self.kd_worst.fit(worst_samples)

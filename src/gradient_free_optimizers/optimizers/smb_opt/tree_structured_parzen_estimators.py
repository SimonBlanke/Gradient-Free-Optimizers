# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .smbo import SMBO

# Use sklearn's KDE if available, otherwise native implementation
try:
    from sklearn.neighbors import KernelDensity

    SKLEARN_AVAILABLE = True
except ImportError:
    from ..._estimators import KernelDensityEstimator as KernelDensity

    SKLEARN_AVAILABLE = False


class TreeStructuredParzenEstimators(SMBO):
    """Tree-structured Parzen Estimator (TPE) optimization algorithm.

    TPE models the conditional probability P(x|y) instead of P(y|x) used by
    other SMBO methods. It maintains two kernel density estimators: one for
    the best-performing samples (l(x)) and one for the rest (g(x)). The
    acquisition function is the ratio l(x)/g(x).

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict, default={"grid": 4, "random": 2, "vertices": 4}
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random iteration.
    nth_process : int, optional
        Process index for parallel optimization.
    warm_start_smbo : pd.DataFrame, optional
        Previous results to initialize the KDEs.
    max_sample_size : int, default=10000000
        Maximum positions to consider.
    sampling : dict or False, default={"random": 1000000}
        Sampling strategy for large search spaces.
    replacement : bool, default=True
        Allow re-evaluation of positions.
    gamma_tpe : float, default=0.2
        Quantile threshold for splitting samples into "good" and "bad".
        Top gamma_tpe fraction are used for l(x), rest for g(x).

    See Also
    --------
    BayesianOptimizer : Gaussian Process based SMBO.
    ForestOptimizer : Tree ensemble based SMBO.
    """

    name = "Tree Structured Parzen Estimators"
    _name_ = "tree_structured_parzen_estimators"
    __name__ = "TreeStructuredParzenEstimators"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=None,
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
        """Split samples into best and worst groups based on gamma_tpe."""
        n_samples = len(self.X_sample)
        n_best = max(round(n_samples * self.gamma_tpe), 1)

        Y_sample = np.array(self.Y_sample)
        index_best = Y_sample.argsort()[-n_best:]
        n_worst = int(n_samples - n_best)
        index_worst = Y_sample.argsort()[:n_worst]

        best_samples = [self.X_sample[i] for i in index_best]
        worst_samples = [self.X_sample[i] for i in index_worst]

        return best_samples, worst_samples

    def _expected_improvement(self):
        """Compute acquisition as ratio of good/bad density estimates."""
        self.pos_comb = self._sampling(self.all_pos_comb)

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

    def _training(self):
        """Fit KDE models on best and worst sample groups."""
        best_samples, worst_samples = self._get_samples()

        self.kd_best.fit(best_samples)
        self.kd_worst.fit(worst_samples)

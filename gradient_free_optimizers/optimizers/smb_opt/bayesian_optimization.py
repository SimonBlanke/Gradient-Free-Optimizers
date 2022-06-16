# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from scipy.stats import norm


from .smbo import SMBO
from .surrogate_models import (
    GPR_linear,
    GPR,
)

gaussian_process = {"gp_nonlinear": GPR(), "gp_linear": GPR_linear()}


def normalize(array):
    num = array - array.min()
    den = array.max() - array.min()

    if den == 0:
        return np.random.random_sample(array.shape)
    else:
        return ((num / den) + 0) / 1


class BayesianOptimizer(SMBO):
    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"

    def __init__(
        self,
        *args,
        gpr=gaussian_process["gp_nonlinear"],
        xi=0.03,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        warnings=100000000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.gpr = gpr
        self.regr = gpr
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.max_sample_size = max_sample_size
        self.sampling = sampling
        self.warnings = warnings

        self.init_warm_start_smbo()

    def _expected_improvement(self):
        all_pos_comb = self._all_possible_pos()
        self.pos_comb = self._sampling(all_pos_comb)

        mu, sigma = self.regr.predict(self.pos_comb, return_std=True)
        # TODO mu_sample = self.regr.predict(self.X_sample)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # with normalization this is always 1
        Y_sample = normalize(np.array(self.Y_sample)).reshape(-1, 1)

        imp = mu - np.max(Y_sample) - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma != 0)

        exploit = imp * norm.cdf(Z)
        explore = sigma * norm.pdf(Z)

        exp_imp = exploit + explore
        exp_imp[sigma == 0.0] = 0.0

        return exp_imp[:, 0]

    def _training(self):
        X_sample = np.array(self.X_sample)
        Y_sample = np.array(self.Y_sample)

        if len(Y_sample) == 0:
            return self.move_random()

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

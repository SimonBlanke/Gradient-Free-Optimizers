# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from scipy.stats import norm


def normalize(array):
    num = array - array.min()
    den = array.max() - array.min()

    if den == 0:
        return np.random.random_sample(array.shape)
    else:
        return ((num / den) + 0) / 1


class ProbabilityOfImprovement:
    def __init__(self, position_l):
        self.position_l = position_l

    def calculate(self, X_sample, Y_sample):
        all_pos_comb = self._all_possible_pos()
        self.pos_comb = self._sampling(all_pos_comb)

        mu, sigma = self.regr.predict(self.pos_comb, return_std=True)
        # TODO mu_sample = self.regr.predict(X_sample)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # with normalization this is always 1
        Y_sample = normalize(np.array(self.Y_sample)).reshape(-1, 1)

        imp = mu - np.max(Y_sample)
        Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma != 0)

        aqu_func = norm.cdf(Z)
        aqu_func[sigma == 0.0] = 0.0

        return aqu_func[:, 0]

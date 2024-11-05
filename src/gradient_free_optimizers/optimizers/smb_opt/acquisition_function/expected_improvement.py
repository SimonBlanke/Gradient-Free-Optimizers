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


class ExpectedImprovement:
    def __init__(self, surrogate_model, position_l, xi):
        self.surrogate_model = surrogate_model
        self.position_l = position_l
        self.xi = xi

    def calculate(self, X_sample, Y_sample):
        mu, sigma = self.surrogate_model.predict(self.position_l, return_std=True)
        # TODO mu_sample = self.surrogate_model.predict(X_sample)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # with normalization this is always 1
        Y_sample = normalize(np.array(Y_sample)).reshape(-1, 1)

        imp = mu - np.max(Y_sample) - self.xi
        Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma != 0)

        exploit = imp * norm.cdf(Z)
        explore = sigma * norm.pdf(Z)

        aqu_func = exploit + explore
        aqu_func[sigma == 0.0] = 0.0

        return aqu_func[:, 0]

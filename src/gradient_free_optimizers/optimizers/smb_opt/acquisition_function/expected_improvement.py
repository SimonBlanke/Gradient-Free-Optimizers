# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from gradient_free_optimizers._array_backend import array, zeros_like, random as np_random
from gradient_free_optimizers._math_backend import norm_cdf, norm_pdf


def normalize(arr):
    arr = array(arr)
    num = arr - arr.min()
    den = arr.max() - arr.min()

    if den == 0:
        return np_random.uniform(0, 1, size=arr.shape)
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
        mu = array(mu).reshape(-1, 1)
        sigma = array(sigma).reshape(-1, 1)

        # with normalization this is always 1
        Y_sample = normalize(array(Y_sample)).reshape(-1, 1)

        imp = mu - Y_sample.max() - self.xi

        # Safe division: divide where sigma != 0, else 0
        Z = zeros_like(sigma)
        for i in range(len(sigma)):
            if sigma[i, 0] != 0:
                Z[i, 0] = imp[i, 0] / sigma[i, 0]

        exploit = imp * norm_cdf(Z)
        explore = sigma * norm_pdf(Z)

        aqu_func = exploit + explore

        # Set acquisition to 0 where sigma == 0
        for i in range(len(sigma)):
            if sigma[i, 0] == 0.0:
                aqu_func[i, 0] = 0.0

        return aqu_func[:, 0]

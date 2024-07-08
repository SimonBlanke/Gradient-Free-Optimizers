import numpy as np
import matplotlib.pyplot as plt

from gradient_free_optimizers import HillClimbingOptimizer


# define the gaussian function for the fit
def gaussian_function(x, A, B, C):
    return A * np.exp(-((x - B) ** 2) / (2 * C**2))


# create the gaussian distributed samples
gauss_np1 = np.random.normal(loc=2, scale=3, size=30000)

bins = 100
min_x = np.min(gauss_np1)
max_x = np.max(gauss_np1)
step_x = (max_x - min_x) / bins

# create the x axis samples
x_range = np.arange(min_x, max_x, step_x)
# the y axis samples to compare with the fitted gaussian
y_gauss_hist = plt.hist(gauss_np1, density=True, bins=bins)[0]


# the objective function for GFO
def fit_gaussian(para):
    A, B, C = para["A"], para["B"], para["C"]
    y_gauss_func = gaussian_function(x_range, A, B, C)

    # compare results of function and hist samples
    diff = np.subtract(y_gauss_func, y_gauss_hist)

    # we want to minimize the difference
    score = -np.abs(diff).sum()
    return score


search_space = {
    "A": np.arange(-10, 10, 0.01),
    "B": np.arange(-10, 10, 0.01),
    "C": np.arange(-10, 10, 0.01),
}

opt = HillClimbingOptimizer(search_space)
opt.search(fit_gaussian, n_iter=10000)

best_parameter = opt.best_para
y_gauss_func_final = gaussian_function(x_range, **best_parameter)

plt.hist(gauss_np1, density=True, bins=bins)
plt.plot(x_range, y_gauss_func_final)
plt.show()

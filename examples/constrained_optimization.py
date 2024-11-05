import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def convex_function(pos_new):
    score = -(pos_new["x1"] * pos_new["x1"] + pos_new["x2"] * pos_new["x2"])
    return score


search_space = {
    "x1": np.arange(-100, 101, 0.1),
    "x2": np.arange(-100, 101, 0.1),
}


def constraint_1(para):
    # only values in 'x1' higher than -5 are valid
    return para["x1"] > -5


# put one or more constraints inside a list
constraints_list = [constraint_1]


# pass list of constraints to the optimizer
opt = RandomSearchOptimizer(search_space, constraints=constraints_list)
opt.search(convex_function, n_iter=50)

search_data = opt.search_data

# the search-data does not contain any samples where x1 is equal or below -5
print("\n search_data \n", search_data, "\n")

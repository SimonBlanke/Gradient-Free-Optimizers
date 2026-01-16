"""Example: Mixed dimension types (discrete, continuous, categorical)."""

import numpy as np

from gradient_free_optimizers import HillClimbingOptimizer


def objective(params):
    """Objective function with mixed parameter types."""
    x = params["x"]  # discrete
    y = params["y"]  # continuous
    algo = params["algorithm"]  # categorical

    # Base score from numeric parameters
    score = -(x**2 + y**2)

    # Bonus based on algorithm choice
    bonus = {"adam": 0.5, "sgd": 0.0, "rmsprop": 0.3}
    score += bonus.get(algo, 0)

    return score


search_space = {
    "x": np.arange(-5, 5, 1),  # Discrete numerical
    "y": (-5.0, 5.0),  # Continuous
    "algorithm": ["adam", "sgd", "rmsprop"],  # Categorical
}

opt = HillClimbingOptimizer(search_space, random_state=1)
opt.search(objective, n_iter=100)

print(f"Best parameters: {opt.best_para}")
print(f"Best score: {opt.best_score:.4f}")

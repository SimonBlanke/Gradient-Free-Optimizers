"""Example: SciPy distribution-backed search space dimensions."""

import math

import numpy as np
from scipy import stats

from gradient_free_optimizers import HillClimbingOptimizer


def objective(params):
    """Optimize mixed parameters with prior-shaped distribution dimensions."""
    learning_rate = params["learning_rate"]
    dropout = params["dropout"]
    n_layers = params["n_layers"]
    activation = params["activation"]

    score = -abs(math.log10(learning_rate) + 3)
    score -= abs(dropout - 0.2)
    score -= 0.1 * abs(n_layers - 3)

    activation_bonus = {"relu": 0.0, "tanh": 0.1, "gelu": 0.3}
    score += activation_bonus[activation]

    return score


search_space = {
    # Sample on a logarithmic scale without building a manual grid.
    "learning_rate": stats.loguniform(1e-5, 1e-1),
    # Use a beta prior for a bounded continuous parameter.
    "dropout": stats.beta(2, 8),
    "n_layers": np.arange(1, 6),
    "activation": ["relu", "tanh", "gelu"],
}

opt = HillClimbingOptimizer(
    search_space,
    initialize={"random": 5},
    random_state=1,
)
opt.search(objective, n_iter=100)

print(f"Best parameters: {opt.best_para}")
print(f"Best score: {opt.best_score:.4f}")

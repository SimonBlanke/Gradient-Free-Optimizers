"""Example: Large search space with automatic vectorization."""

import time

import numpy as np

from gradient_free_optimizers import HillClimbingOptimizer
from gradient_free_optimizers._dimension_iterator import VECTORIZATION_THRESHOLD


def objective(params):
    """Sum of squares for high-dimensional optimization."""
    return -sum(v**2 for v in params.values())


# Create a search space with 1500 dimensions (above vectorization threshold)
n_dims = 1500
search_space = {f"x{i}": np.linspace(-5, 5, 20) for i in range(n_dims)}

opt = HillClimbingOptimizer(search_space)

print(f"Dimensions: {opt.conv.n_dimensions}")
print(f"Vectorization threshold: {VECTORIZATION_THRESHOLD}")
print(f"Using vectorization: {opt._can_vectorize()}")

start = time.time()
opt.search(objective, n_iter=10000, verbosity=["progress_bar"])
elapsed = time.time() - start

print(f"\nCompleted in {elapsed:.2f}s")
print(f"Best score: {opt.best_score:.2f}")

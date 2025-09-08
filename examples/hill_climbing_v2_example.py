from dataclasses import dataclass

import numpy as np
import scipy.stats as st

from gradient_free_optimizers import HillClimbingOptimizer
from gradient_free_optimizers._search_space.base import BaseSearchSpace


@dataclass
class MySpace(BaseSearchSpace):
    # Continuous real
    x: tuple = (-5.0, 5.0)
    # Integer range
    k: tuple = (1, 10, "[)")
    # Distribution (bounded log-uniform)
    lr: object = st.loguniform(1e-4, 1e-1)
    # Categorical with mixed types (str, bool)
    act: list = ("relu", "tanh", "gelu")
    use_bn: list = (True, False)
    # Fixed
    seed: int = 7


def objective(params):
    # Simple synthetic objective combining different types
    x = params["x"]
    k = int(round(params["k"]))
    lr = float(params["lr"])  # positive
    act = params["act"]
    use_bn = params["use_bn"]

    # Base: inverted quadratic on x, with preference for middle k
    score = -(x**2) - (k - 5) ** 2 * 0.1
    # Prefer middling learning rates
    score -= (np.log10(lr) + 2.5) ** 2
    # Small categorical tweaks
    score += {"relu": 0.0, "tanh": 0.2, "gelu": 0.4}[act]
    score += 0.3 if use_bn else 0.0
    return score


def main():
    space = MySpace()
    opt = HillClimbingOptimizer(
        search_space=space,
        initialize={"grid": 3, "random": 3, "vertices": 2},
        random_state=123,
        epsilon=0.1,
        n_neighbours=3,
    )

    opt.search(objective, n_iter=25, verbosity=["print_results"])


if __name__ == "__main__":
    main()

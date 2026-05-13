"""Example: Compare boundary strategies for out-of-bounds candidates."""

from gradient_free_optimizers import HillClimbingOptimizer

BOUNDARIES = ("clip", "reflect", "periodic", "random", "intermediate")


def objective(params):
    """Optimize a target close to the search-space boundary."""
    x = params["x"]
    y = params["y"]
    return -((x - 0.95) ** 2 + (y - 0.05) ** 2)


search_space = {
    "x": (0.0, 1.0),
    "y": (0.0, 1.0),
}

initialize = {
    "warm_start": [{"x": 0.5, "y": 0.5}],
    "random": 2,
}

for boundary in BOUNDARIES:
    opt = HillClimbingOptimizer(
        search_space,
        boundary=boundary,
        epsilon=0.6,
        n_neighbours=8,
        initialize=initialize,
        random_state=7,
    )
    opt.search(objective, n_iter=80, verbosity=[])

    best = opt.best_para
    print(
        f"{boundary:>12}: "
        f"x={best['x']:.3f}, y={best['y']:.3f}, score={opt.best_score:.4f}"
    )

===================
Boundary Strategies
===================

This example compares the ``boundary`` strategies available on optimizers.
The objective has its optimum close to the edge of the search space, and the
large ``epsilon`` value makes Hill Climbing generate candidates that often
leave the valid range.

.. code-block:: python

    from gradient_free_optimizers import HillClimbingOptimizer

    BOUNDARIES = ("clip", "reflect", "periodic", "random", "intermediate")


    def objective(params):
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
            f"x={best['x']:.3f}, y={best['y']:.3f}, "
            f"score={opt.best_score:.4f}"
        )

Use ``"clip"`` for the conservative default, ``"reflect"`` to bounce
overshooting steps back into the space, ``"periodic"`` for cyclic domains,
``"random"`` for extra exploration, and ``"intermediate"`` for damped
edge handling.

The same code is available as ``examples/boundary_strategies.py``.

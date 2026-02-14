# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Forest optimizer using tree ensemble surrogate models."""

from typing import Literal

from .._init_utils import get_default_initialize, get_default_sampling
from ..optimizers import ForestOptimizer as _ForestOptimizer
from ..search import Search


class ForestOptimizer(_ForestOptimizer, Search):
    r"""
    Sequential model-based optimizer using tree ensemble surrogate models.

    Forest Optimizer uses tree-based ensemble models (Random Forest or Extra
    Trees) as surrogate models instead of Gaussian Processes. This approach
    scales better to high-dimensional problems and large datasets while
    providing uncertainty estimates through the variance of tree predictions.

    The algorithm follows the same sequential model-based optimization framework:
    (1) fit a tree ensemble to observed data, (2) use the ensemble to predict
    mean and variance at candidate points, (3) select the next point using an
    acquisition function, and (4) update the model with new observations.

    The algorithm is well-suited for:

    - High-dimensional optimization problems (>20 dimensions)
    - Problems with many observations where GP fitting becomes slow
    - Categorical or mixed parameter spaces
    - Situations where tree-based models naturally fit the problem structure

    Tree ensembles handle categorical variables naturally and can capture
    non-smooth objective functions better than GPs in some cases.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore, defined as a dictionary mapping parameter
        names to arrays of possible values.

        Each key is a parameter name (string), and each value is a numpy array
        or list of discrete values that the parameter can take. The optimizer
        will only evaluate positions that are on this discrete grid.

        Example: A 2D search space with 100 points per dimension::

            search_space = {
                "x": np.linspace(-10, 10, 100),
                "y": np.linspace(-10, 10, 100),
            }

        The resolution of each dimension (number of points in the array)
        directly affects optimization quality and speed. More points give
        finer resolution but increase the search space size exponentially.
    initialize : dict[str, int], default={"vertices": 4, "random": 2}
        Strategy for generating initial positions before the main optimization
        loop begins. Initialization samples are evaluated first, and the best
        one becomes the starting point for the optimizer.

        Supported keys:

        - ``"grid"``: ``int`` -- Number of positions on a regular grid.
        - ``"vertices"``: ``int`` -- Number of corner/edge positions of the
          search space.
        - ``"random"``: ``int`` -- Number of uniformly random positions.
        - ``"warm_start"``: ``list[dict]`` -- Specific positions to evaluate,
          each as a dict mapping parameter names to values.

        Multiple strategies can be combined::

            initialize = {"vertices": 4, "random": 10}
            initialize = {"warm_start": [{"x": 0.5, "y": 1.0}], "random": 5}

        More initialization samples improve the starting point but consume
        iterations from ``n_iter``. For expensive objectives, a few targeted
        warm-start points are often more efficient than many random samples.
    constraints : list[callable], default=[]
        A list of constraint functions that restrict the search space. Each
        constraint is a callable that receives a parameter dictionary and
        returns ``True`` if the position is valid, ``False`` if it should
        be rejected.

        Rejected positions are discarded and regenerated: the optimizer
        resamples a new candidate position (up to 100 retries per step).
        During initialization, positions that violate constraints are
        filtered out entirely.

        Example: Constrain the search to a circular region::

            def circular_constraint(para):
                return para["x"]**2 + para["y"]**2 <= 25

            constraints = [circular_constraint]

        Multiple constraints are combined with AND logic (all must return
        ``True``).
    random_state : int or None, default=None
        Seed for the random number generator to ensure reproducible results.

        - ``None``: Use a new random state each run (non-deterministic).
        - ``int``: Seed the random number generator for reproducibility.

        Setting a fixed seed is recommended for debugging and benchmarking.
        Different seeds may lead to different optimization trajectories,
        especially for stochastic optimizers.
    rand_rest_p : float, default=0
        Probability of performing a random restart instead of the normal
        algorithm step. At each iteration, a uniform random number is drawn;
        if it falls below ``rand_rest_p``, the optimizer jumps to a random
        position instead of following its strategy.

        - ``0.0``: No random restarts (pure algorithm behavior).
        - ``0.01-0.05``: Light diversification, helps escape shallow local
          optima.
        - ``0.1-0.3``: Aggressive restarts, useful for highly multi-modal
          landscapes.
        - ``1.0``: Equivalent to random search.

        This is especially useful for local search optimizers (Hill Climbing,
        Simulated Annealing) that can get trapped. For population-based
        optimizers, the effect is less pronounced since they already maintain
        diversity through multiple agents.
    warm_start_smbo : object or None, default=None
        Previous SMBO state for warm-starting the surrogate model. Allows
        continuing optimization from a previous run by reusing the fitted
        model state, avoiding the cost of refitting from scratch.

        Pass ``None`` to start fresh without warm-starting.
    max_sample_size : int, default=10000000
        Maximum number of candidate points to consider when optimizing the
        acquisition function. The surrogate model predicts scores for these
        candidates, and the best one according to the acquisition function
        is selected for evaluation.

        Larger values improve acquisition optimization quality but increase
        memory usage and computation time. For most problems, the default
        is more than sufficient. Reduce this if memory is a concern.
    sampling : dict, default={"random": 1000000}
        Configuration for how candidate points are generated for acquisition
        function optimization. The key specifies the sampling strategy and
        the value specifies the number of samples.

        Currently supported: ``{"random": N}`` for uniform random sampling
        of N candidate points from the search space.

        Example::

            sampling = {"random": 500000}  # Fewer candidates, faster
    replacement : bool, default=True
        Whether to sample candidate points with replacement when generating
        candidates for acquisition function optimization. When ``True``,
        the same point can appear multiple times. When ``False``, each
        candidate is unique, ensuring diversity but potentially slower for
        very large sample sizes.
    tree_regressor : {"extra_tree", "random_forest"}, default="extra_tree"
        The type of tree ensemble used as the surrogate model.

        - ``"extra_tree"``: Extra-Trees (Extremely Randomized Trees).
          Faster to train because it uses random split thresholds
          instead of searching for optimal splits. Provides smoother
          uncertainty estimates. Recommended for most cases.
        - ``"random_forest"``: Standard Random Forest. Uses optimal
          split searching, which can provide more accurate predictions
          but is slower to train.
    tree_para : dict, default={"n_estimators": 100}
        Parameters passed directly to the underlying scikit-learn tree
        regressor. Common options include:

        - ``n_estimators``: Number of trees in the ensemble. More trees
          provide better uncertainty estimates but increase training time.
          Default is 100.
        - ``max_depth``: Maximum tree depth. Shallower trees provide
          smoother predictions.
        - ``min_samples_split``: Minimum samples to split a node.

        Example::

            tree_para = {"n_estimators": 200, "max_depth": 10}
    xi : float, default=0.03
        Exploration-exploitation trade-off parameter for the acquisition
        function. Controls how much the optimizer values uncertain
        regions (high variance across trees) over predicted-good regions.

        - ``0.0``: Pure exploitation, samples where the ensemble predicts
          the best score.
        - ``0.01-0.05``: Mild exploration (default region).
        - ``0.1-0.3``: Moderate exploration, favors uncertain regions.

        Same role as ``xi`` in BayesianOptimizer, but uncertainty is
        estimated from the variance across tree predictions rather than
        from a Gaussian Process.

    Notes
    -----
    The algorithm follows the same SMBO framework as Bayesian Optimization
    but replaces the Gaussian Process with a tree ensemble:

    1. Fit a tree ensemble (Extra-Trees or Random Forest) to all observed
       ``(position, score)`` pairs.
    2. For each candidate point, predict mean (average across trees) and
       uncertainty (variance across trees).
    3. Select the next point using the acquisition function weighted by
       ``xi``.

    Tree ensembles offer several advantages over GPs:

    - **Scalability**: Training is :math:`O(n \\log n)` vs. :math:`O(n^3)`
      for GPs.
    - **Categorical support**: Trees handle categorical features natively.
    - **Non-stationarity**: Trees can model functions with varying
      smoothness across the space.

    The trade-off is that tree-based uncertainty estimates are less
    principled than GP posterior variance.

    For visual explanations and tuning guides, see
    the :ref:`Forest Optimizer user guide <forest>`.

    See Also
    --------
    BayesianOptimizer : SMBO using Gaussian Processes.
    TreeStructuredParzenEstimators : SMBO using density ratio estimation.
    EnsembleOptimizer : Combines multiple surrogate model types.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import ForestOptimizer

    >>> def high_dim_function(para):
    ...     return -sum(para[f"x{i}"] ** 2 for i in range(5))

    >>> search_space = {f"x{i}": np.linspace(-5, 5, 50) for i in range(5)}

    >>> opt = ForestOptimizer(
    ...     search_space,
    ...     tree_regressor="extra_tree",
    ...     tree_para={"n_estimators": 50},
    ... )
    >>> opt.search(high_dim_function, n_iter=200)
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = None,
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        warm_start_smbo=None,
        max_sample_size: int = 10000000,
        sampling: dict[Literal["random"], int] = None,
        replacement: bool = True,
        tree_regressor="extra_tree",
        tree_para={"n_estimators": 100},
        xi=0.03,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []
        if sampling is None:
            sampling = get_default_sampling()

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
            tree_regressor=tree_regressor,
            tree_para=tree_para,
            xi=xi,
        )

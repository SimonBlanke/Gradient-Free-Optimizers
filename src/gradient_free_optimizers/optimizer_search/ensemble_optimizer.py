# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Ensemble optimizer combining multiple surrogate model types."""

from typing import Literal

from .._init_utils import get_default_initialize, get_default_sampling
from ..optimizers import EnsembleOptimizer as _EnsembleOptimizer
from ..search import Search


class EnsembleOptimizer(_EnsembleOptimizer, Search):
    """
    Sequential model-based optimizer combining multiple surrogate model types.

    Ensemble Optimizer combines predictions from multiple surrogate model types
    to make more robust optimization decisions. By aggregating different models
    (e.g., Gaussian Processes, tree ensembles, kernel density estimators), the
    ensemble can leverage the strengths of each approach while mitigating their
    individual weaknesses.

    This approach is particularly useful when it is unclear which surrogate model
    type is best suited for a given problem. The ensemble provides more stable
    uncertainty estimates and can adapt to different regions of the search space
    where different model types may perform better.

    The algorithm is well-suited for:

    - Problems where the best surrogate model type is unknown
    - Robust optimization requiring reliable uncertainty estimates
    - Complex objective functions with varying characteristics
    - Situations where model selection overhead is acceptable

    The ensemble combines SMBO techniques to provide a general-purpose
    optimizer that performs well across a wide range of problem types.

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

    Notes
    -----
    The Ensemble Optimizer combines predictions from multiple SMBO
    surrogate models:

    1. Each constituent model (GP, tree ensemble, KDE) is fitted to the
       observed data independently.
    2. Predictions are aggregated across models, providing more robust
       mean and uncertainty estimates than any single model.
    3. The next evaluation point is selected based on the combined
       acquisition function.

    This approach is related to model stacking and mixture-of-experts,
    applied to the surrogate modeling step of Bayesian optimization.
    It is most beneficial when the objective function has characteristics
    that are captured differently by different model types (e.g., smooth
    regions suited to GPs and non-smooth regions suited to trees).

    For visual explanations and tuning guides, see
    the :ref:`Ensemble Optimizer user guide <ensemble>`.

    See Also
    --------
    BayesianOptimizer : SMBO using a single Gaussian Process surrogate.
    TreeStructuredParzenEstimators : SMBO using kernel density estimation.
    ForestOptimizer : SMBO using a single tree ensemble surrogate.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import EnsembleOptimizer

    >>> def complex_objective(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(x**2 + y**2 + 0.5 * np.sin(10 * x) * np.cos(10 * y))

    >>> search_space = {
    ...     "x": np.linspace(-2, 2, 100),
    ...     "y": np.linspace(-2, 2, 100),
    ... }

    >>> opt = EnsembleOptimizer(search_space)
    >>> opt.search(complex_objective, n_iter=100)
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
        )

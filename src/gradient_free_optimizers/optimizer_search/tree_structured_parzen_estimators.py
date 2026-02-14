# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Tree-structured Parzen Estimator (TPE) using kernel density estimation."""

from typing import Literal

from .._init_utils import get_default_initialize, get_default_sampling
from ..optimizers import (
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
)
from ..search import Search


class TreeStructuredParzenEstimators(_TreeStructuredParzenEstimators, Search):
    r"""
    Sequential model-based optimizer using kernel density estimation.

    Tree-structured Parzen Estimator (TPE) is an efficient sequential model-based
    optimization algorithm that models the probability of good and bad parameter
    configurations separately. Unlike Bayesian Optimization which models P(y|x),
    TPE models P(x|y) by maintaining two density estimators: one for parameters
    that led to good results (l) and one for poor results (g).

    The algorithm selects the next point by maximizing the ratio l(x)/g(x), which
    is equivalent to optimizing Expected Improvement but is computationally more
    efficient. TPE uses kernel density estimation (Parzen estimators) to model
    these distributions, with a tree structure for handling conditional parameters.

    The algorithm is well-suited for:

    - Hyperparameter optimization of machine learning models
    - High-dimensional optimization problems (scales better than GP-based methods)
    - Problems with conditional or hierarchical parameter spaces
    - Situations requiring fast surrogate model updates

    The `gamma_tpe` parameter controls the quantile threshold for splitting
    observations into good and bad groups. A value of 0.2 means the top 20%
    of observations are considered "good".

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
    gamma_tpe : float, default=0.2
        Quantile threshold for splitting observations into "good" and
        "bad" groups. Observations with scores in the top ``gamma_tpe``
        fraction are modeled by the "good" density :math:`l(x)`, and the
        rest by the "bad" density :math:`g(x)`.

        - ``0.05-0.1``: Very selective, only the best ~5-10% are
          considered "good". Stronger exploitation, faster convergence.
        - ``0.15-0.25``: Moderate selectivity (default region). Good
          balance for most problems.
        - ``0.3-0.5``: Lenient, up to half the observations are "good".
          More exploration, slower convergence.

        With few observations (< 20), the split may result in very small
        groups. TPE handles this gracefully through kernel density
        smoothing.

    Notes
    -----
    TPE models the probability distributions of good and bad parameter
    configurations separately:

    1. Sort all observations by score and split at the ``gamma_tpe``
       quantile into good set :math:`D_l` and bad set :math:`D_g`.
    2. Fit kernel density estimators :math:`l(x)` and :math:`g(x)` to
       each set using Parzen (kernel) estimation.
    3. Select the next point by maximizing:

    .. math::

        \\frac{l(x)}{g(x)}

    This is equivalent to maximizing Expected Improvement but avoids
    directly modeling :math:`P(y|x)`. The density ratio :math:`l(x)/g(x)`
    is high for configurations that look like good observations and
    unlike bad ones.

    TPE scales better than GP-based Bayesian Optimization because kernel
    density estimation is :math:`O(n)` rather than :math:`O(n^3)`.

    For visual explanations and tuning guides, see
    the :ref:`TPE user guide <tpe>`.

    See Also
    --------
    BayesianOptimizer : SMBO using Gaussian Processes.
    ForestOptimizer : SMBO using tree ensembles.
    EnsembleOptimizer : Combines multiple surrogate types for robustness.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import TreeStructuredParzenEstimators

    >>> def ml_hyperparameter_objective(para):
    ...     # Simulating ML model performance
    ...     learning_rate = para["learning_rate"]
    ...     n_estimators = para["n_estimators"]
    ...     return -(0.9 - abs(learning_rate - 0.1) - abs(n_estimators - 100) / 1000)

    >>> search_space = {
    ...     "learning_rate": np.logspace(-4, 0, 100),
    ...     "n_estimators": np.arange(10, 500, 10),
    ... }

    >>> opt = TreeStructuredParzenEstimators(search_space, gamma_tpe=0.25)
    >>> opt.search(ml_hyperparameter_objective, n_iter=100)
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
        gamma_tpe=0.2,
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
            gamma_tpe=gamma_tpe,
        )

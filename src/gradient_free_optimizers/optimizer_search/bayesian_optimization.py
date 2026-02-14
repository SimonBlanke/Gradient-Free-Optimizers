# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Bayesian optimization using Gaussian Process surrogate models."""

from typing import Literal

from .._init_utils import get_default_initialize, get_default_sampling
from ..optimizers import BayesianOptimizer as _BayesianOptimizer
from ..search import Search


class BayesianOptimizer(_BayesianOptimizer, Search):
    r"""
    Sequential model-based optimizer using Gaussian Process surrogate models.

    Bayesian Optimization is a powerful technique for optimizing expensive
    black-box functions. It builds a probabilistic surrogate model (Gaussian
    Process) of the objective function and uses an acquisition function to
    determine the most promising points to evaluate next. This approach is
    sample-efficient, requiring fewer function evaluations than many other
    methods to find good solutions.

    The algorithm works by: (1) fitting a Gaussian Process to observed data,
    (2) using the GP to predict mean and uncertainty at unobserved points,
    (3) selecting the next point to evaluate based on an acquisition function
    (Expected Improvement), and (4) updating the model with the new observation.

    The algorithm is well-suited for:

    - Expensive objective functions (e.g., ML model training, simulations)
    - Low to moderate dimensional problems (typically < 20 dimensions)
    - Problems where sample efficiency is critical
    - Hyperparameter optimization of machine learning models

    The `xi` parameter controls the exploration-exploitation trade-off in the
    Expected Improvement acquisition function. Higher values encourage more
    exploration of uncertain regions.

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
    gpr : object or None, default=None
        The Gaussian Process Regressor used as the surrogate model.
        Accepts three forms:

        - ``None``: Uses the default GPR implementation.
        - **Class**: A GPR class that will be instantiated automatically.
        - **Instance**: A pre-configured GPR instance.

        Custom GPR implementations should follow the scikit-learn API with
        ``fit(X, y)`` and ``predict(X, return_std=True)`` methods. This
        allows using different kernels, noise models, or entirely
        different GP libraries.
    xi : float, default=0.03
        Exploration-exploitation trade-off parameter for the Expected
        Improvement (EI) acquisition function. Controls how much the
        optimizer values uncertain regions over predicted-good regions.

        - ``0.0``: Pure exploitation, always samples where the GP predicts
          the best score. Tends to converge fast but may miss the global
          optimum.
        - ``0.01-0.05``: Mild exploration (default region). Good balance
          for most problems.
        - ``0.1-0.3``: Moderate exploration, favors uncertain regions.
        - ``1.0+``: Strong exploration, heavily prefers unexplored areas.

        Higher ``xi`` is useful early in optimization or when the search
        space is large relative to the number of evaluations.

    Notes
    -----
    The algorithm follows a sequential loop:

    1. Fit a Gaussian Process to all observed ``(position, score)`` pairs.
    2. For each candidate point, predict mean :math:`\\mu(x)` and
       standard deviation :math:`\\sigma(x)`.
    3. Compute Expected Improvement:

    .. math::

        EI(x) = (\\mu(x) - f_{\\text{best}} - \\xi) \\cdot
        \\Phi(Z) + \\sigma(x) \\cdot \\phi(Z)

    where :math:`Z = (\\mu(x) - f_{\\text{best}} - \\xi) / \\sigma(x)`,
    :math:`\\Phi` is the standard normal CDF, and :math:`\\phi` is the
    standard normal PDF.

    4. Evaluate the point with the highest EI.

    The GP surrogate provides a principled balance between exploration
    (sampling where :math:`\\sigma(x)` is high) and exploitation (sampling
    where :math:`\\mu(x)` is high). Fitting the GP has complexity
    :math:`O(n^3)` where n is the number of observations, which can become
    a bottleneck for large evaluation budgets.

    For visual explanations and tuning guides, see
    the :ref:`Bayesian Optimization user guide <bayesian>`.

    See Also
    --------
    ForestOptimizer : SMBO using tree ensembles, scales better.
    TreeStructuredParzenEstimators : SMBO using kernel density estimation.
    EnsembleOptimizer : Combines multiple surrogate model types for robustness.
    LipschitzOptimizer : Global optimization using Lipschitz continuity bounds.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import BayesianOptimizer

    >>> def expensive_function(para):
    ...     # Simulating an expensive evaluation
    ...     x, y = para["x"], para["y"]
    ...     return -((x - 0.5) ** 2 + (y - 0.5) ** 2)

    >>> search_space = {
    ...     "x": np.linspace(0, 1, 100),
    ...     "y": np.linspace(0, 1, 100),
    ... }

    >>> opt = BayesianOptimizer(search_space, xi=0.01)
    >>> opt.search(expensive_function, n_iter=50)
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
        gpr=None,
        xi: float = 0.03,
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
            gpr=gpr,
            xi=xi,
        )

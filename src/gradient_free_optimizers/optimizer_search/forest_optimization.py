# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Forest optimizer using tree ensemble surrogate models."""

from typing import Literal

from .._init_utils import get_default_initialize, get_default_sampling
from ..optimizers_new import ForestOptimizer as _ForestOptimizer
from ..search import Search


class ForestOptimizer(_ForestOptimizer, Search):
    """
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
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the the search process.
    warm_start_smbo : object, optional
        Previous SMBO state for warm starting optimization.
    max_sample_size : int
        Maximum number of candidate points for acquisition optimization.
        Default is 10000000.
    sampling : dict
        Configuration for candidate sampling. Default is {"random": 1000000}.
    replacement : bool
        Whether to sample candidates with replacement. Default is True.
    tree_regressor : str
        The tree ensemble type to use. Options are "extra_tree" for Extra Trees
        or "random_forest" for Random Forest. Default is "extra_tree".
    tree_para : dict
        Parameters passed to the tree regressor. Common options include
        n_estimators, max_depth, etc. Default is {"n_estimators": 100}.
    xi : float
        Exploration-exploitation trade-off parameter for the acquisition
        function. Higher values favor exploration. Default is 0.03.

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

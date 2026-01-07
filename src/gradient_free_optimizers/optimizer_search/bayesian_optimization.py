# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal, Union

from ..search import Search
from ..optimizers import BayesianOptimizer as _BayesianOptimizer
from ..optimizers.smb_opt.bayesian_optimization import gaussian_process


class BayesianOptimizer(_BayesianOptimizer, Search):
    """
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
        The probability of a random iteration during the search process.
    warm_start_smbo : object, optional
        Previous SMBO state for warm starting. Allows continuing optimization
        from a previous run.
    max_sample_size : int
        Maximum number of candidate points to consider when optimizing the
        acquisition function. Default is 10000000.
    sampling : dict
        Configuration for candidate sampling. Default is {"random": 1000000}.
    replacement : bool
        Whether to sample candidates with replacement. Default is True.
    gpr : object
        The Gaussian Process Regressor configuration. Uses a nonlinear GP
        by default, suitable for most optimization problems.
    xi : float
        Exploration-exploitation trade-off parameter for Expected Improvement.
        Higher values favor exploration of uncertain regions. Default is 0.03.

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
        search_space: Dict[str, list],
        initialize: Dict[
            Literal["grid", "vertices", "random", "warm_start"],
            Union[int, list[dict]],
        ] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[callable] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        warm_start_smbo=None,
        max_sample_size: int = 10000000,
        sampling: Dict[Literal["random"], int] = {"random": 1000000},
        replacement: bool = True,
        gpr=gaussian_process["gp_nonlinear"],
        xi: float = 0.03,
    ):
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

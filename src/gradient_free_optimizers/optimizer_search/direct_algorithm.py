# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""DIRECT algorithm using adaptive hyperrectangle subdivision."""

from typing import Literal

import pandas as pd

from .._init_utils import get_default_initialize
from ..optimizers_new import DirectAlgorithm as _DirectAlgorithm
from ..search import Search


class DirectAlgorithm(_DirectAlgorithm, Search):
    """
    Deterministic global optimizer using adaptive hyperrectangle subdivision.

    DIRECT (DIviding RECTangles) is a deterministic global optimization algorithm
    that systematically divides the search space into smaller hyperrectangles
    and samples their centers. The algorithm identifies "potentially optimal"
    rectangles based on a trade-off between the function value at the center
    and the size of the rectangle, balancing local refinement and global
    exploration without requiring derivatives or Lipschitz constants.

    Note: Unlike surrogate-model-based optimizers (Bayesian, Forest, TPE),
    DIRECT does not train a model. It uses deterministic subspace division
    with Lipschitz bounds for selection.

    At each iteration, DIRECT identifies hyperrectangles that could contain the
    global optimum (based on comparing function values and rectangle sizes),
    then divides these rectangles along their longest dimension. This creates
    a tree structure that adaptively refines the search in promising regions
    while maintaining coverage of the entire space.

    The algorithm is well-suited for:

    - Global optimization requiring deterministic guarantees
    - Lipschitz continuous functions (but doesn't require knowing the constant)
    - Low to moderate dimensional problems (typically < 10 dimensions)
    - Problems where both local and global search are important

    DIRECT provides a balance between exploration (large rectangles) and
    exploitation (rectangles with good function values) through its selection
    criterion, making it robust without requiring parameter tuning.

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
    warm_start : pd.DataFrame, optional
        Previous optimization results to warm-start the algorithm.
    resolution : int
        Number of grid points for continuous dimensions. Default is 100.
        Continuous dimensions (specified as tuples like (0.0, 10.0)) are
        automatically discretized into this many evenly-spaced points.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import DirectAlgorithm

    >>> def multimodal(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(np.sin(x) * np.sin(y) + 0.1 * (x**2 + y**2))

    >>> search_space = {
    ...     "x": np.linspace(-3, 3, 100),
    ...     "y": np.linspace(-3, 3, 100),
    ... }

    >>> opt = DirectAlgorithm(search_space)
    >>> opt.search(multimodal, n_iter=200)
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
        warm_start: pd.DataFrame = None,
        resolution: int = 100,
        # Legacy SMBO parameters - no-op, kept for backwards compatibility
        warm_start_smbo: pd.DataFrame = None,
        max_sample_size: int = 10000000,
        sampling: dict[str, int] = None,
        replacement: bool = True,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start=warm_start,
            resolution=resolution,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

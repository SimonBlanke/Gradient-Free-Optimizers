# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
from functools import wraps

from ..._array_backend import rint, clip, array, random as np_random
from ..._math_backend import cdist

from .search_tracker import SearchTracker
from .converter import Converter
from .init_positions import Initializer

from .utils import set_random_seed, move_random


def _get_dist_func(name):
    """Get distribution function from array backend."""
    dist_map = {
        "normal": np_random.normal,
        "laplace": np_random.laplace,
        "logistic": np_random.logistic,
        "gumbel": np_random.gumbel,
    }
    return dist_map[name]


class CoreOptimizer(SearchTracker):
    """
    Core optimization mechanics for position generation and evaluation.

    CoreOptimizer provides the fundamental building blocks that all
    optimization algorithms use: position tracking, search space conversion,
    initialization handling, and movement utilities. It bridges the gap
    between abstract optimization logic and concrete array manipulations.

    This class manages:

    - **Position Tracking**: Via inheritance from SearchTracker, maintains
      current, new, and best positions with their scores
    - **Search Space Conversion**: The ``conv`` (Converter) object handles
      transformations between positions (array indices), values (actual
      parameter values), and parameters (dict format)
    - **Initialization**: The ``init`` (Initializer) object generates
      starting positions based on the initialization strategy
    - **Movement Utilities**: Methods like ``move_climb`` and ``move_random``
      for generating candidate positions

    The position representation uses integer indices into the search space
    arrays, which enables efficient constraint checking and discrete
    optimization. The Converter handles mapping these to actual values.

    Parameters
    ----------
    search_space : dict[str, array-like]
        Dictionary mapping parameter names to arrays of possible values.
    initialize : dict
        Initialization configuration passed to Initializer.
    constraints : list[callable]
        Constraint functions passed to Converter.
    random_state : int or None
        Random seed for reproducibility.
    rand_rest_p : float
        Probability of random restart (used by ``random_iteration`` decorator).
    nth_process : int or None
        Process identifier for parallel scenarios.

    Attributes
    ----------
    conv : Converter
        Handles position/value/parameter conversions and constraint checking.
    init : Initializer
        Generates initial positions based on the initialization strategy.
    nth_init : int
        Counter for initialization steps completed.
    nth_trial : int
        Counter for total evaluations (init + iterations).
    search_state : str
        Either "init" (initialization phase) or "iter" (iteration phase).

    See Also
    --------
    SearchTracker : Tracks positions and scores throughout optimization.
    Converter : Handles search space transformations.
    Initializer : Generates initial positions.
    """

    def __init__(
        self,
        search_space,
        initialize,
        constraints,
        random_state,
        rand_rest_p,
        nth_process,
    ):
        super().__init__()

        self.search_space = search_space
        self.initialize = initialize
        self.constraints = constraints if constraints is not None else []
        self.random_state = random_state
        self.rand_rest_p = rand_rest_p
        self.nth_process = nth_process

        self.random_seed = set_random_seed(self.nth_process, self.random_state)

        self.conv = Converter(self.search_space, self.constraints)
        self.init = Initializer(self.conv, self.initialize)

        self.nth_init = 0
        self.nth_trial = 0
        self.search_state = "init"

    def random_iteration(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.rand_rest_p > random.uniform(0, 1):
                return self.move_random()
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def move_climb(
        self, pos, epsilon=0.03, distribution="normal", epsilon_mod=1
    ):
        dist_func = _get_dist_func(distribution)
        while True:
            sigma = self.conv.max_positions * epsilon * epsilon_mod
            pos_normal = dist_func(pos, sigma, pos.shape)
            pos = self.conv2pos(pos_normal)

            if self.conv.not_in_constraint(pos):
                return pos
            epsilon_mod *= 1.01

    def conv2pos(self, pos):
        # position to int
        r_pos = rint(pos)

        n_zeros = [0] * len(self.conv.max_positions)
        # clip into search space boundaries
        pos = clip(r_pos, n_zeros, self.conv.max_positions).astype(int)

        r_pos_2d = array(r_pos).reshape((1, -1))
        pos_2d = array(pos).reshape((1, -1))
        dist = cdist(r_pos_2d, pos_2d)
        threshold = self.conv.search_space_size / (100**self.conv.n_dimensions)

        if dist[0][0] > threshold:
            return self.move_random()

        return pos

    def move_random(self):
        while True:
            pos = move_random(self.conv.search_space_positions)
            if self.conv.not_in_constraint(pos):
                return pos

    @SearchTracker.track_new_pos
    def init_pos(self):
        init_pos = self.init.init_positions_l[self.nth_init]
        return init_pos

    @SearchTracker.track_new_score
    def evaluate_init(self, score_new):
        if self.pos_best is None:
            self.pos_best = self.pos_new
            self.score_best = score_new

        if self.pos_current is None:
            self.pos_current = self.pos_new
            self.score_current = score_new

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Particle swarm optimization inspired by collective swarm behavior."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import ParticleSwarmOptimizer as _ParticleSwarmOptimizer
from ..search import Search


class ParticleSwarmOptimizer(_ParticleSwarmOptimizer, Search):
    r"""
    Swarm intelligence optimizer inspired by collective behavior of bird flocks.

    Particle Swarm Optimization (PSO) is a population-based metaheuristic that
    simulates the social behavior of birds flocking or fish schooling. Each
    particle in the swarm represents a candidate solution that moves through
    the search space influenced by its own best-known position (cognitive
    component) and the swarm's best-known position (social component).

    The velocity update equation balances three components: inertia (tendency
    to continue in the current direction), cognitive attraction (pull toward
    personal best), and social attraction (pull toward global best). This
    creates emergent optimization behavior where particles explore promising
    regions while sharing information about good solutions.

    The algorithm is well-suited for:

    - Continuous optimization problems
    - Multimodal functions where multiple regions need exploration
    - Problems where gradient information is unavailable
    - Real-time optimization where quick convergence is needed

    The balance between `cognitive_weight` and `social_weight` controls
    exploration vs exploitation. Higher cognitive weight promotes individual
    exploration, while higher social weight accelerates convergence toward
    the best-known solution.

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
    population : int, default=10
        Number of particles in the swarm. Each particle represents a candidate
        solution that moves through the search space.

        - ``5-10``: Small populations, fast per generation but risk of
          premature convergence.
        - ``15-30``: Good diversity-convergence balance for most problems.
        - ``50-100``: Thorough exploration, better for high-dimensional or
          highly multimodal problems.

        Each individual requires one function evaluation per generation, so
        total cost scales linearly with population size. As a rule of thumb,
        use larger populations for higher-dimensional or more multimodal
        problems.
    inertia : float, default=0.5
        Weight applied to a particle's current velocity, controlling
        momentum. Determines the tendency to continue moving in the same
        direction.

        - ``0.2-0.4``: Low inertia, particles change direction easily.
          Better exploitation, faster convergence.
        - ``0.5-0.7``: Balanced momentum (default region).
        - ``0.8-0.9``: High inertia, particles resist direction changes.
          Better exploration, slower convergence.

        Classic PSO literature recommends starting around 0.9 and
        decreasing to 0.4, but this implementation uses a fixed value.

    cognitive_weight : float, default=0.5
        Attraction strength toward each particle's personal best position.
        Controls the "memory" component of particle movement.

        - ``0.0``: No personal memory, particles ignore their own history.
        - ``0.5``: Moderate personal attraction (default).
        - ``1.5-2.0``: Strong personal attraction, promotes individual
          exploration of each particle's promising region.

        Higher values make particles circle around their own best
        positions more tightly.

    social_weight : float, default=0.5
        Attraction strength toward the swarm's global best position.
        Controls the "social" or information-sharing component.

        - ``0.0``: No social influence, particles ignore the swarm's best.
        - ``0.5``: Moderate social attraction (default).
        - ``1.5-2.0``: Strong social attraction, promotes rapid
          convergence toward the best-known position.

        Higher ``social_weight`` relative to ``cognitive_weight`` causes
        faster but potentially premature convergence.

    temp_weight : float, default=0.2
        Temperature-like randomness weight added to the velocity update.
        Introduces stochastic perturbation to help particles escape local
        optima.

        - ``0.0``: No random perturbation (deterministic velocity update).
        - ``0.1-0.3``: Mild randomness, small perturbations (default
          region).
        - ``0.5-1.0``: Strong randomness, aggressive exploration.

    Notes
    -----
    Each particle's velocity and position are updated at each iteration:

    .. math::

        v_{t+1} = w \\cdot v_t
        + c_1 \\cdot r_1 \\cdot (p_{\\text{best}} - x_t)
        + c_2 \\cdot r_2 \\cdot (g_{\\text{best}} - x_t)
        + c_3 \\cdot r_3

    .. math::

        x_{t+1} = x_t + v_{t+1}

    where :math:`w` is ``inertia``, :math:`c_1` is ``cognitive_weight``,
    :math:`c_2` is ``social_weight``, :math:`c_3` is ``temp_weight``,
    :math:`r_1, r_2, r_3` are random vectors, :math:`p_{\\text{best}}`
    is the particle's personal best, and :math:`g_{\\text{best}}` is the
    global best.

    Time complexity per iteration is O(``population`` * d), where d is
    the number of dimensions.

    For visual explanations and tuning guides, see
    the :ref:`Particle Swarm Optimization user guide <particle_swarm>`.

    See Also
    --------
    SpiralOptimization : Population-based search using spiral trajectories.
    DifferentialEvolutionOptimizer : Evolution using vector differences.
    GeneticAlgorithmOptimizer : Evolution using crossover and mutation.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import ParticleSwarmOptimizer

    >>> def rastrigin(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

    >>> search_space = {
    ...     "x": np.linspace(-5.12, 5.12, 100),
    ...     "y": np.linspace(-5.12, 5.12, 100),
    ... }

    >>> opt = ParticleSwarmOptimizer(
    ...     search_space, population=20, inertia=0.7,
    ...     cognitive_weight=1.5, social_weight=1.5,
    ... )
    >>> opt.search(rastrigin, n_iter=500)
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
        population: int = 10,
        inertia: float = 0.5,
        cognitive_weight: float = 0.5,
        social_weight: float = 0.5,
        temp_weight: float = 0.2,
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
            population=population,
            inertia=inertia,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            temp_weight=temp_weight,
        )

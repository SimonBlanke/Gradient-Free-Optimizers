# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Differential evolution using vector differences for mutation."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer,
)
from ..search import Search


class DifferentialEvolutionOptimizer(_DifferentialEvolutionOptimizer, Search):
    r"""
    Evolutionary optimizer using vector differences for mutation.

    Differential Evolution (DE) is a powerful population-based optimizer
    particularly effective for continuous optimization problems. The key
    innovation is using weighted differences between population members to
    generate mutations, which automatically adapts the search scale to the
    current population distribution.

    For each individual, DE creates a trial vector by adding a weighted
    difference of two random population members to a third member (mutation),
    then applying crossover with the original individual. If the trial vector
    improves upon the original, it replaces it in the next generation. This
    simple yet effective scheme provides robust global optimization.

    The algorithm is well-suited for:

    - Continuous optimization problems with real-valued parameters
    - Multimodal functions with many local optima
    - Non-separable problems where parameters interact
    - Black-box optimization without gradient information

    DE is known for its simplicity, few control parameters, and robust
    performance across a wide range of problems. The `mutation_rate` (often
    called F) controls the amplification of differential variation.

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
        Number of individuals in the population. DE requires at least 4
        individuals for the differential mutation operator to work.

        - ``5-10``: Small populations, fast per generation but risk of
          premature convergence.
        - ``15-30``: Good diversity-convergence balance for most problems.
        - ``50-100``: Thorough exploration, better for high-dimensional or
          highly multimodal problems.

        Each individual requires one function evaluation per generation, so
        total cost scales linearly with population size. As a rule of thumb,
        use larger populations for higher-dimensional or more multimodal
        problems.
    mutation_rate : float, default=0.9
        Scaling factor F for the differential mutation vector. Controls
        the amplification of the difference between two random population
        members.

        - ``0.2-0.4``: Small mutations, conservative search. Better
          for functions with narrow basins.
        - ``0.5-0.7``: Moderate mutations, good balance.
        - ``0.8-1.0``: Large mutations, strong exploration (default
          region). Robust choice for unknown landscapes.
        - ``>1.0``: Very large mutations, may overshoot but can help
          escape deep local optima.

        The literature often denotes this as F. A common robust choice
        is F = 0.8.

    crossover_rate : float, default=0.9
        Probability CR that each parameter in the trial vector inherits
        from the mutant vector instead of the original individual.

        - ``0.1-0.3``: Mostly retains the original, only a few parameters
          mutated per trial. Good for separable problems.
        - ``0.5``: Balanced crossover.
        - ``0.8-1.0``: Most parameters come from the mutant, creating
          very different trial vectors (default region). Good for
          non-separable problems.

        At least one parameter is always taken from the mutant to ensure
        each trial vector differs from its parent.

    Notes
    -----
    Differential Evolution uses the DE/rand/1 mutation scheme:

    For each individual :math:`x_i`, a mutant vector is created:

    .. math::

        v_i = x_{r1} + F \\cdot (x_{r2} - x_{r3})

    where :math:`r1, r2, r3` are distinct random population members and
    :math:`F` is the ``mutation_rate``. A trial vector :math:`u_i` is
    then formed by crossover:

    .. math::

        u_{i,j} = \\begin{cases}
        v_{i,j} & \\text{if } \\text{rand}() < CR \\text{ or } j = j_{\\text{rand}} \\\\
        x_{i,j} & \\text{otherwise}
        \\end{cases}

    If :math:`f(u_i) > f(x_i)`, the trial vector replaces the original
    in the next generation. This greedy selection ensures monotonic
    population improvement.

    For visual explanations and tuning guides, see
    the :ref:`Differential Evolution user guide <differential_evolution>`.

    See Also
    --------
    EvolutionStrategyOptimizer : Evolutionary approach using Gaussian mutation.
    GeneticAlgorithmOptimizer : Evolutionary crossover and mutation.
    ParticleSwarmOptimizer : Population-based swarm intelligence approach.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import DifferentialEvolutionOptimizer

    >>> def rosenbrock(para):
    ...     x, y = para["x"], para["y"]
    ...     return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ... }

    >>> opt = DifferentialEvolutionOptimizer(
    ...     search_space, population=20, mutation_rate=0.8, crossover_rate=0.9
    ... )
    >>> opt.search(rosenbrock, n_iter=500)
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
        population=10,
        mutation_rate=0.9,
        crossover_rate=0.9,
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
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

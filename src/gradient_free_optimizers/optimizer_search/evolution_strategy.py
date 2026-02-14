# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Evolution strategy using self-adaptive mutation for continuous domains."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import (
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
)
from ..search import Search


class EvolutionStrategyOptimizer(_EvolutionStrategyOptimizer, Search):
    """
    Evolutionary optimizer focused on self-adaptive mutation for continuous domains.

    Evolution Strategy (ES) is an evolutionary algorithm originally designed
    for continuous parameter optimization. Unlike genetic algorithms that
    emphasize crossover, ES primarily relies on mutation as the main variation
    operator. The algorithm generates offspring by adding random perturbations
    to parent solutions, then selects the best individuals for the next
    generation.

    Two main selection schemes exist: (mu, lambda) where only offspring compete
    for selection (replace_parents=True), and (mu + lambda) where parents and
    offspring compete together (replace_parents=False). The comma strategy
    provides stronger selection pressure and better escapes from local optima,
    while the plus strategy preserves good solutions.

    The algorithm is well-suited for:

    - Continuous optimization problems
    - Real-valued parameter tuning
    - Problems where fine-grained mutation control is beneficial
    - Situations requiring self-adaptive step sizes

    The `mutation_rate` controls the probability of perturbing each parameter,
    while `crossover_rate` determines how often recombination is applied.

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
        Number of parent individuals (mu) in the population. Parents are
        the elite solutions that survive selection.

        - ``5-10``: Small populations, fast per generation but risk of
          premature convergence.
        - ``15-30``: Good diversity-convergence balance for most problems.
        - ``50-100``: Thorough exploration, better for high-dimensional or
          highly multimodal problems.

        Each individual requires one function evaluation per generation, so
        total cost scales linearly with population size. As a rule of thumb,
        use larger populations for higher-dimensional or more multimodal
        problems.
    offspring : int, default=20
        Number of offspring (lambda) generated each generation through
        mutation and optional crossover. Should typically be larger than
        ``population`` for effective selection pressure.

        - ``population``: Minimal offspring, weak selection. Common
          notation: (mu, mu) or (mu + mu).
        - ``2-5 * population``: Standard range. Notation examples:
          (10, 20) or (10, 50).
        - ``10 * population``: Very strong selection, only the best
          survive.

        The ratio ``offspring / population`` determines selection
        pressure. A ratio of 7:1 is commonly recommended in the
        literature.

    replace_parents : bool, default=False
        Selection scheme controlling how the next generation is formed.

        - ``False``: **(mu + lambda)** strategy. Parents compete with
          offspring for survival. Preserves elite solutions, which
          guarantees monotonic improvement. More conservative.
        - ``True``: **(mu, lambda)** strategy. Only offspring can become
          parents. Provides stronger selection pressure and better ability
          to escape local optima, but may lose good solutions. Requires
          ``offspring >= population``.

    mutation_rate : float, default=0.7
        Probability of mutating each parameter in an offspring. ES
        traditionally relies heavily on mutation as the primary variation
        operator.

        - ``0.3-0.5``: Moderate mutation, stable evolution.
        - ``0.7``: Standard ES mutation rate (default).
        - ``0.9-1.0``: Nearly all genes mutated, maximum exploration.

    crossover_rate : float, default=0.3
        Probability of applying recombination to create offspring.
        In classical ES, crossover plays a secondary role compared to
        mutation.

        - ``0.0``: Pure mutation-based ES (classical approach).
        - ``0.3``: Mild crossover (default).
        - ``0.5-0.7``: Stronger recombination, more GA-like behavior.

    Notes
    -----
    Evolution Strategy follows a (mu, lambda) or (mu + lambda) scheme:

    1. Generate ``offspring`` new solutions by mutating and optionally
       recombining ``population`` parents.
    2. Evaluate all offspring.
    3. Select the best ``population`` individuals as parents for the
       next generation (from offspring only in comma strategy, or from
       parents + offspring in plus strategy).

    The key difference from Genetic Algorithms is the emphasis on
    mutation as the primary search operator, making ES naturally suited
    for continuous optimization. Classical ES uses self-adaptive mutation
    step sizes, though this implementation uses a fixed ``mutation_rate``.

    For visual explanations and tuning guides, see
    the :ref:`Evolution Strategy user guide <evolution_strategy>`.

    See Also
    --------
    GeneticAlgorithmOptimizer : Crossover-focused evolutionary approach.
    DifferentialEvolutionOptimizer : Mutation using vector differences.
    ParticleSwarmOptimizer : Swarm intelligence without evolutionary operators.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import EvolutionStrategyOptimizer

    >>> def sphere(para):
    ...     return -(para["x"] ** 2 + para["y"] ** 2 + para["z"] ** 2)

    >>> search_space = {
    ...     "x": np.linspace(-5, 5, 100),
    ...     "y": np.linspace(-5, 5, 100),
    ...     "z": np.linspace(-5, 5, 100),
    ... }

    >>> opt = EvolutionStrategyOptimizer(
    ...     search_space, population=15, offspring=30, replace_parents=True
    ... )
    >>> opt.search(sphere, n_iter=500)
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
        offspring=20,
        replace_parents=False,
        mutation_rate=0.7,
        crossover_rate=0.3,
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
            offspring=offspring,
            replace_parents=replace_parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

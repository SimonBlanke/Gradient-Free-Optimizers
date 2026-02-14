# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Genetic algorithm using selection, crossover, and mutation operators."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer
from ..search import Search


class GeneticAlgorithmOptimizer(_GeneticAlgorithmOptimizer, Search):
    """
    Evolutionary optimizer inspired by natural selection and genetics.

    Genetic Algorithm (GA) is a population-based metaheuristic that mimics
    the process of natural evolution. The algorithm maintains a population of
    candidate solutions (individuals) that evolve over generations through
    selection, crossover (recombination), and mutation operators. Better
    solutions have higher probability of surviving and reproducing, gradually
    improving the population's fitness.

    Each generation follows these steps: (1) Selection - choosing parents based
    on fitness, (2) Crossover - combining parent genes to create offspring,
    (3) Mutation - introducing random changes for diversity, and (4) Replacement -
    forming the next generation from parents and offspring.

    The algorithm is well-suited for:

    - Combinatorial and discrete optimization problems
    - Multimodal optimization landscapes
    - Problems where the solution can be encoded as a chromosome-like structure
    - Situations requiring robust global search

    The balance between `crossover_rate` and `mutation_rate` controls
    exploration vs exploitation. Higher crossover promotes combining good
    solutions, while higher mutation maintains population diversity.

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
        Number of individuals in the population. Each individual is a
        candidate solution whose genes (parameters) evolve over generations.

        - ``5-10``: Small populations, fast per generation but risk of
          premature convergence.
        - ``15-30``: Good diversity-convergence balance for most problems.
        - ``50-100``: Thorough exploration, better for high-dimensional or
          highly multimodal problems.

        Each individual requires one function evaluation per generation, so
        total cost scales linearly with population size. As a rule of thumb,
        use larger populations for higher-dimensional or more multimodal
        problems.
    offspring : int, default=10
        Number of offspring to generate each generation through crossover
        and mutation. Typically equal to or larger than the population
        size.

        - ``population/2``: Few offspring, conservative evolution.
        - ``population``: Standard generational replacement (default).
        - ``2*population``: Many offspring, stronger selection pressure.

        More offspring provide more candidates for selection but increase
        computational cost per generation.

    crossover : str, default="discrete-recombination"
        The crossover operator for combining parent genes into offspring.

        - ``"discrete-recombination"``: Each offspring gene is randomly
          chosen from one of the parents. Simple and effective for most
          problems.

    n_parents : int, default=2
        Number of parents selected for each crossover operation. Standard
        genetic algorithms use 2 parents, but multi-parent crossover can
        increase diversity.

        - ``2``: Standard two-parent crossover (default).
        - ``3-5``: Multi-parent crossover, increases genetic diversity.

    mutation_rate : float, default=0.5
        Probability of mutating each gene (parameter) in an offspring.
        Mutation introduces random changes to maintain population
        diversity and prevent premature convergence.

        - ``0.01-0.1``: Low mutation, preserves good solutions but risks
          stagnation.
        - ``0.2-0.5``: Moderate mutation, good exploration-exploitation
          balance.
        - ``0.7-1.0``: High mutation, strong exploration but may disrupt
          good building blocks.

    crossover_rate : float, default=0.5
        Probability of applying crossover to create an offspring vs.
        cloning a parent directly. Controls the balance between
        recombination and pure selection.

        - ``0.1-0.3``: Mostly cloning, crossover is rare. Useful for
          problems where recombination is disruptive.
        - ``0.5``: Balanced (default).
        - ``0.7-1.0``: Frequent crossover, promotes combination of good
          building blocks from different parents.

        Higher ``crossover_rate`` with lower ``mutation_rate`` emphasizes
        recombination; the inverse emphasizes mutation-driven exploration.

    Notes
    -----
    Each generation follows this cycle:

    1. **Selection**: Parents are chosen from the population based on
       fitness (higher scores are preferred).
    2. **Crossover**: With probability ``crossover_rate``, pairs of
       parents recombine their genes to produce offspring.
    3. **Mutation**: Each gene in each offspring is randomly perturbed
       with probability ``mutation_rate``.
    4. **Replacement**: The best individuals from the combined parent
       and offspring pool form the next generation.

    The algorithm relies on the "building block hypothesis": good
    partial solutions (schemata) are combined through crossover to
    construct better complete solutions over generations.

    For visual explanations and tuning guides, see
    the :ref:`Genetic Algorithm user guide <genetic_algorithm>`.

    See Also
    --------
    EvolutionStrategyOptimizer : Mutation-focused evolutionary approach.
    DifferentialEvolutionOptimizer : Mutation using vector differences.
    ParticleSwarmOptimizer : Swarm-based optimization.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import GeneticAlgorithmOptimizer

    >>> def knapsack_like(para):
    ...     return para["item1"] * 10 + para["item2"] * 20 + para["item3"] * 15

    >>> search_space = {
    ...     "item1": np.array([0, 1]),
    ...     "item2": np.array([0, 1]),
    ...     "item3": np.array([0, 1]),
    ... }

    >>> opt = GeneticAlgorithmOptimizer(
    ...     search_space, population=20, mutation_rate=0.3, crossover_rate=0.7
    ... )
    >>> opt.search(knapsack_like, n_iter=100)
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
        offspring=10,
        crossover="discrete-recombination",
        n_parents=2,
        mutation_rate=0.5,
        crossover_rate=0.5,
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
            crossover=crossover,
            n_parents=n_parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

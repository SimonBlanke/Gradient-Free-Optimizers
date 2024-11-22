# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import List, Dict, Literal

from .search import Search
from .optimizers import (
    HillClimbingOptimizer as _HillClimbingOptimizer,
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer as _RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer as _DownhillSimplexOptimizer,
    RandomSearchOptimizer as _RandomSearchOptimizer,
    GridSearchOptimizer as _GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer as _RandomAnnealingOptimizer,
    PowellsMethod as _PowellsMethod,
    PatternSearch as _PatternSearch,
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer,
    ParticleSwarmOptimizer as _ParticleSwarmOptimizer,
    SpiralOptimization as _SpiralOptimization,
    GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer,
    BayesianOptimizer as _BayesianOptimizer,
    LipschitzOptimizer as _LipschitzOptimizer,
    DirectAlgorithm as _DirectAlgorithm,
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
    ForestOptimizer as _ForestOptimizer,
    EnsembleOptimizer as _EnsembleOptimizer,
)


class HillClimbingOptimizer(_HillClimbingOptimizer, Search):
    """
    **Hill climbing** is a very basic optimization technique, that explores the search space only localy. It starts at an initial point, which is often chosen randomly and continues to move to positions within its neighbourhood with a better solution. It has no method against getting stuck in local optima.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. Formatted as a dictionary with parameter
        names as keys and a list of possible values as values.
    initialize : dict[str, int]
        The method to generate initial positions. Formatted as a dictionary with
        a single key of one of ["grid", "random", "vertices"] and the number of
        points to generate as the value.
    constraints : list[callable]
        A list of constraints. Each constraint is a dictionary with the key
        "fun" and the value a function that takes a numpy array of size
        (n_points, n_dimensions) and returns a boolean array of size n_points
        indicating which points are valid.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of randomly re-starting the search process.
    nth_process : int
        The process number of this optimizer. Used for seeding the random state.
    epsilon : float
        The step-size for the climbing.
    distribution : str
        The type of distribution to sample from.
    n_neighbours : int
        The number of neighbours to sample and evaluate before moving to the best
        of those neighbours.
    """

    def __init__(
        self,
        search_space: Dict[str, list],
        initialize: Dict[str, int] = {"grid": 4, "random": 2, "vertices": 4},
        constraints: List[Dict[str, callable]] = [],
        random_state: int = None,
        rand_rest_p: float = 0,
        nth_process: int = None,
        epsilon: float = 0.03,
        distribution: Literal[
            "normal", "laplace", "gumbel", "logistic"
        ] = "normal",
        n_neighbours: int = 3,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            epsilon=epsilon,
            distribution=distribution,
            n_neighbours=n_neighbours,
        )


class StochasticHillClimbingOptimizer(_StochasticHillClimbingOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RepulsingHillClimbingOptimizer(_RepulsingHillClimbingOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimulatedAnnealingOptimizer(_SimulatedAnnealingOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DownhillSimplexOptimizer(_DownhillSimplexOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomSearchOptimizer(_RandomSearchOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GridSearchOptimizer(_GridSearchOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomRestartHillClimbingOptimizer(
    _RandomRestartHillClimbingOptimizer, Search
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomAnnealingOptimizer(_RandomAnnealingOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PowellsMethod(_PowellsMethod, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PatternSearch(_PatternSearch, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ParallelTemperingOptimizer(_ParallelTemperingOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ParticleSwarmOptimizer(_ParticleSwarmOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpiralOptimization(_SpiralOptimization, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GeneticAlgorithmOptimizer(_GeneticAlgorithmOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EvolutionStrategyOptimizer(_EvolutionStrategyOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DifferentialEvolutionOptimizer(_DifferentialEvolutionOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BayesianOptimizer(_BayesianOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LipschitzOptimizer(_LipschitzOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DirectAlgorithm(_DirectAlgorithm, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TreeStructuredParzenEstimators(_TreeStructuredParzenEstimators, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ForestOptimizer(_ForestOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EnsembleOptimizer(_EnsembleOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

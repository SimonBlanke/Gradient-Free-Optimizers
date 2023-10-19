# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

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
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer,
    BayesianOptimizer as _BayesianOptimizer,
    LipschitzOptimizer as _LipschitzOptimizer,
    DirectAlgorithm as _DirectAlgorithm,
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators,
    ForestOptimizer as _ForestOptimizer,
    EnsembleOptimizer as _EnsembleOptimizer,
)


class HillClimbingOptimizer(_HillClimbingOptimizer, Search):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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


class RandomRestartHillClimbingOptimizer(_RandomRestartHillClimbingOptimizer, Search):
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


class EvolutionStrategyOptimizer(_EvolutionStrategyOptimizer, Search):
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

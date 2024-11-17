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
from ._inherit_signature import inherit_signature


class HillClimbingOptimizer(_HillClimbingOptimizer, Search):
    @inherit_signature(_HillClimbingOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StochasticHillClimbingOptimizer(_StochasticHillClimbingOptimizer, Search):
    @inherit_signature(_StochasticHillClimbingOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RepulsingHillClimbingOptimizer(_RepulsingHillClimbingOptimizer, Search):
    @inherit_signature(_RepulsingHillClimbingOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimulatedAnnealingOptimizer(_SimulatedAnnealingOptimizer, Search):
    @inherit_signature(_SimulatedAnnealingOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DownhillSimplexOptimizer(_DownhillSimplexOptimizer, Search):
    @inherit_signature(_DownhillSimplexOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomSearchOptimizer(_RandomSearchOptimizer, Search):
    @inherit_signature(_RandomSearchOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GridSearchOptimizer(_GridSearchOptimizer, Search):
    @inherit_signature(_GridSearchOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomRestartHillClimbingOptimizer(
    _RandomRestartHillClimbingOptimizer, Search
):
    @inherit_signature(_RandomRestartHillClimbingOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomAnnealingOptimizer(_RandomAnnealingOptimizer, Search):
    @inherit_signature(_RandomAnnealingOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PowellsMethod(_PowellsMethod, Search):
    @inherit_signature(_PowellsMethod.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PatternSearch(_PatternSearch, Search):
    @inherit_signature(_PatternSearch.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ParallelTemperingOptimizer(_ParallelTemperingOptimizer, Search):
    @inherit_signature(_ParallelTemperingOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ParticleSwarmOptimizer(_ParticleSwarmOptimizer, Search):
    @inherit_signature(_ParticleSwarmOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpiralOptimization(_SpiralOptimization, Search):
    @inherit_signature(_SpiralOptimization.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GeneticAlgorithmOptimizer(_GeneticAlgorithmOptimizer, Search):
    @inherit_signature(_GeneticAlgorithmOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EvolutionStrategyOptimizer(_EvolutionStrategyOptimizer, Search):
    @inherit_signature(_EvolutionStrategyOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DifferentialEvolutionOptimizer(_DifferentialEvolutionOptimizer, Search):
    @inherit_signature(_DifferentialEvolutionOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BayesianOptimizer(_BayesianOptimizer, Search):
    @inherit_signature(_BayesianOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LipschitzOptimizer(_LipschitzOptimizer, Search):
    @inherit_signature(_LipschitzOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DirectAlgorithm(_DirectAlgorithm, Search):
    @inherit_signature(_DirectAlgorithm.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TreeStructuredParzenEstimators(_TreeStructuredParzenEstimators, Search):
    @inherit_signature(_TreeStructuredParzenEstimators.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ForestOptimizer(_ForestOptimizer, Search):
    @inherit_signature(_ForestOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EnsembleOptimizer(_EnsembleOptimizer, Search):
    @inherit_signature(_EnsembleOptimizer.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

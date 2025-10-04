# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ._optimizers.local_opt import (
    HillClimbingOptimizer as _HillClimbingOptimizer_,
    StochasticHillClimbingOptimizer as _StochasticHillClimbingOptimizer_,
    RepulsingHillClimbingOptimizer as _RepulsingHillClimbingOptimizer_,
    SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer_,
    DownhillSimplexOptimizer as _DownhillSimplexOptimizer_,
)

from ._optimizers.global_opt import (
    RandomSearchOptimizer as _RandomSearchOptimizer_,
    RandomRestartHillClimbingOptimizer as _RandomRestartHillClimbingOptimizer_,
    PowellsMethod as _PowellsMethod_,
    PatternSearch as _PatternSearch_,
    LipschitzOptimizer as _LipschitzOptimizer_,
    DirectAlgorithm as _DirectAlgorithm_,
)


from ._optimizers.pop_opt import (
    ParallelTemperingOptimizer as _ParallelTemperingOptimizer_,
    ParticleSwarmOptimizer as _ParticleSwarmOptimizer_,
    SpiralOptimization as _SpiralOptimization_,
    GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer_,
    EvolutionStrategyOptimizer as _EvolutionStrategyOptimizer_,
    DifferentialEvolutionOptimizer as _DifferentialEvolutionOptimizer_,
)

from ._optimizers.smb_opt import (
    BayesianOptimizer as _BayesianOptimizer_,
    TreeStructuredParzenEstimators as _TreeStructuredParzenEstimators_,
    ForestOptimizer as _ForestOptimizer_,
)

from ._optimizers.exp_opt import (
    RandomAnnealingOptimizer as _RandomAnnealingOptimizer_,
    EnsembleOptimizer as _EnsembleOptimizer_,
)

from ._optimizers.grid import (
    GridSearchOptimizer as _GridSearchOptimizer_,
)

from ._ask_tell_base_optimizer import AskTellOptimizer


# Create ask/tell wrapper classes for each optimizer
class HillClimbingOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _HillClimbingOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class StochasticHillClimbingOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _StochasticHillClimbingOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class RepulsingHillClimbingOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _RepulsingHillClimbingOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class SimulatedAnnealingOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _SimulatedAnnealingOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class DownhillSimplexOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _DownhillSimplexOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class RandomSearchOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        nth_process=None,
        **kwargs,
    ):
        super().__init__(
            _RandomSearchOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class GridSearchOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _GridSearchOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class RandomRestartHillClimbingOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _RandomRestartHillClimbingOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class PowellsMethod(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _PowellsMethod_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class PatternSearch(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _PatternSearch_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class LipschitzOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _LipschitzOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class DirectAlgorithm(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _DirectAlgorithm_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class RandomAnnealingOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _RandomAnnealingOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class ParallelTemperingOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _ParallelTemperingOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class ParticleSwarmOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _ParticleSwarmOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class SpiralOptimization(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _SpiralOptimization_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class GeneticAlgorithmOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _GeneticAlgorithmOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class EvolutionStrategyOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _EvolutionStrategyOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class DifferentialEvolutionOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _DifferentialEvolutionOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class BayesianOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _BayesianOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class TreeStructuredParzenEstimators(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _TreeStructuredParzenEstimators_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class ForestOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _ForestOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


class EnsembleOptimizer(AskTellOptimizer):
    def __init__(
        self,
        search_space,
        random_state=None,
        constraints=None,
        **kwargs,
    ):
        super().__init__(
            _EnsembleOptimizer_,
            search_space,
            random_state=random_state,
            constraints=constraints,
            **kwargs,
        )


__all__ = [
    "HillClimbingOptimizer",
    "StochasticHillClimbingOptimizer",
    "RepulsingHillClimbingOptimizer",
    "SimulatedAnnealingOptimizer",
    "DownhillSimplexOptimizer",
    "RandomSearchOptimizer",
    "GridSearchOptimizer",
    "RandomRestartHillClimbingOptimizer",
    "PowellsMethod",
    "PatternSearch",
    "LipschitzOptimizer",
    "DirectAlgorithm",
    "RandomAnnealingOptimizer",
    "ParallelTemperingOptimizer",
    "ParticleSwarmOptimizer",
    "SpiralOptimization",
    "GeneticAlgorithmOptimizer",
    "EvolutionStrategyOptimizer",
    "DifferentialEvolutionOptimizer",
    "BayesianOptimizer",
    "TreeStructuredParzenEstimators",
    "ForestOptimizer",
    "EnsembleOptimizer",
]

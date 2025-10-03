# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .local_opt import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
)

from .global_opt import (
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    PowellsMethod,
    PatternSearch,
    LipschitzOptimizer,
    DirectAlgorithm,
)


from .pop_opt import (
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer,
)

from .smb_opt import (
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)

from .exp_opt import (
    RandomAnnealingOptimizer,
    EnsembleOptimizer,
)

from .grid import (
    GridSearchOptimizer,
)

from .ask_tell_optimizer import AskTellOptimizer

# Create ask/tell wrapper classes for each optimizer
class HillClimbing(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(HillClimbingOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class StochasticHillClimbing(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(StochasticHillClimbingOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class RepulsingHillClimbing(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(RepulsingHillClimbingOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class SimulatedAnnealing(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(SimulatedAnnealingOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class DownhillSimplex(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(DownhillSimplexOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class RandomSearch(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(RandomSearchOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class GridSearch(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(GridSearchOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class RandomRestartHillClimbing(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(RandomRestartHillClimbingOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class Powell(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(PowellsMethod, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class Pattern(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(PatternSearch, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class Lipschitz(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(LipschitzOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class Direct(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(DirectAlgorithm, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class RandomAnnealing(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(RandomAnnealingOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class ParallelTempering(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(ParallelTemperingOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class ParticleSwarm(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(ParticleSwarmOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class Spiral(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(SpiralOptimization, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class GeneticAlgorithm(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(GeneticAlgorithmOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class EvolutionStrategy(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(EvolutionStrategyOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class DifferentialEvolution(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(DifferentialEvolutionOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class BayesianOptimization(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(BayesianOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class TPE(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(TreeStructuredParzenEstimators, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class Forest(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(ForestOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

class Ensemble(AskTellOptimizer):
    def __init__(self, search_space, init_positions, init_scores, random_state=None, constraints=None, **kwargs):
        super().__init__(EnsembleOptimizer, search_space, init_positions, init_scores,
                         random_state=random_state, constraints=constraints, **kwargs)

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
    # Ask/Tell Interface
    "AskTellOptimizer",
    "HillClimbing",
    "StochasticHillClimbing",
    "RepulsingHillClimbing",
    "SimulatedAnnealing",
    "DownhillSimplex",
    "RandomSearch",
    "GridSearch",
    "RandomRestartHillClimbing",
    "Powell",
    "Pattern",
    "Lipschitz",
    "Direct",
    "RandomAnnealing",
    "ParallelTempering",
    "ParticleSwarm",
    "Spiral",
    "GeneticAlgorithm",
    "EvolutionStrategy",
    "DifferentialEvolution",
    "BayesianOptimization",
    "TPE",
    "Forest",
    "Ensemble",
]

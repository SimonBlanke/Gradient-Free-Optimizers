# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.metadata

__version__ = importlib.metadata.version("gradient-free-optimizers")
__license__ = "MIT"

from .optimizer_search import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    PowellsMethod,
    PatternSearch,
    LipschitzOptimizer,
    DirectAlgorithm,
    RandomAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
    EnsembleOptimizer,
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

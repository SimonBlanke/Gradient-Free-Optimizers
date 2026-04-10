# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.metadata

__version__ = importlib.metadata.version("gradient-free-optimizers")
__license__ = "MIT"

from ._fitness_mapper import FitnessMapper, ScalarIdentity, WeightedSum
from ._result import ObjectiveResult
from .optimizer_search import (
    BayesianOptimizer,
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    DirectAlgorithm,
    DownhillSimplexOptimizer,
    EvolutionStrategyOptimizer,
    ForestOptimizer,
    GeneticAlgorithmOptimizer,
    GridSearchOptimizer,
    HillClimbingOptimizer,
    LipschitzOptimizer,
    MOEADOptimizer,
    NSGA2Optimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomAnnealingOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    SMSEMOAOptimizer,
    SpiralOptimization,
    StochasticHillClimbingOptimizer,
    TreeStructuredParzenEstimators,
)

__all__ = [
    "ObjectiveResult",
    "FitnessMapper",
    "ScalarIdentity",
    "WeightedSum",
    "NSGA2Optimizer",
    "MOEADOptimizer",
    "SMSEMOAOptimizer",
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
    "CMAESOptimizer",
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
]

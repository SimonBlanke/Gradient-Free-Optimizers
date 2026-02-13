# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Optimizer module with Template Method Pattern architecture.

This module provides optimizers with explicit dimension-type support
and vectorized batch operations for high-dimensional optimization.

See ARCHITECTURE.md for detailed documentation.
"""

# Base classes
from .base_optimizer import BaseOptimizer
from .core_optimizer import CoreOptimizer

# Experimental optimizers
from .exp_opt import (
    EnsembleOptimizer,
    RandomAnnealingOptimizer,
)

# Global optimizers
from .global_opt import (
    DirectAlgorithm,
    LipschitzOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
)

# Grid search optimizers
from .grid import (
    DiagonalGridSearch,
    GridSearchOptimizer,
    OrthogonalGridSearch,
)

# Local optimizers
from .local_opt import (
    DownhillSimplexOptimizer,
    HillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticHillClimbingOptimizer,
)

# Population-based optimizers
from .pop_opt import (
    BasePopulationOptimizer,
    DifferentialEvolutionOptimizer,
    EvolutionStrategyOptimizer,
    GeneticAlgorithmOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
)

# Sequential Model-Based optimizers
from .smb_opt import (
    SMBO,
    BayesianOptimizer,
    ForestOptimizer,
    TreeStructuredParzenEstimators,
)

__all__ = [
    # Base
    "BaseOptimizer",
    "CoreOptimizer",
    # Local
    "HillClimbingOptimizer",
    "SimulatedAnnealingOptimizer",
    "StochasticHillClimbingOptimizer",
    "RepulsingHillClimbingOptimizer",
    "DownhillSimplexOptimizer",
    # Global
    "RandomSearchOptimizer",
    "RandomRestartHillClimbingOptimizer",
    "PatternSearch",
    "PowellsMethod",
    "DirectAlgorithm",
    "LipschitzOptimizer",
    # Population
    "BasePopulationOptimizer",
    "ParticleSwarmOptimizer",
    "DifferentialEvolutionOptimizer",
    "GeneticAlgorithmOptimizer",
    "EvolutionStrategyOptimizer",
    "SpiralOptimization",
    "ParallelTemperingOptimizer",
    # SMBO
    "SMBO",
    "BayesianOptimizer",
    "ForestOptimizer",
    "TreeStructuredParzenEstimators",
    # Grid
    "GridSearchOptimizer",
    "DiagonalGridSearch",
    "OrthogonalGridSearch",
    # Experimental
    "RandomAnnealingOptimizer",
    "EnsembleOptimizer",
]

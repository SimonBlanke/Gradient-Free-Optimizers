# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Ask/tell interface for Gradient-Free-Optimizers.

Provides the same optimizer algorithms as the top-level package, but with
a batch-capable ask/tell interface instead of the managed search() loop.

Usage::

    from gradient_free_optimizers.ask_tell import HillClimbingOptimizer

    opt = HillClimbingOptimizer(
        search_space,
        initial_evaluations=[
            ({"x": 0.5, "y": 1.0}, 0.8),
            ({"x": -3.0, "y": 0.0}, 0.2),
        ],
    )

    for _ in range(25):
        params_list = opt.ask(n=4)
        scores = [my_function(p) for p in params_list]
        opt.tell(scores)

    print(opt.best_para, opt.best_score)
"""

from .bayesian_optimization import BayesianOptimizer
from .cma_es import CMAESOptimizer
from .differential_evolution import DifferentialEvolutionOptimizer
from .direct_algorithm import DirectAlgorithm
from .downhill_simplex import DownhillSimplexOptimizer
from .evolution_strategy import EvolutionStrategyOptimizer
from .forest_optimization import ForestOptimizer
from .genetic_algorithm import GeneticAlgorithmOptimizer
from .grid_search import GridSearchOptimizer
from .hill_climbing import HillClimbingOptimizer
from .lipschitz_optimizer import LipschitzOptimizer
from .parallel_tempering import ParallelTemperingOptimizer
from .particle_swarm_optimization import ParticleSwarmOptimizer
from .pattern_search import PatternSearch
from .powells_method import PowellsMethod
from .random_annealing import RandomAnnealingOptimizer
from .random_restart_hill_climbing import RandomRestartHillClimbingOptimizer
from .random_search import RandomSearchOptimizer
from .repulsing_hill_climbing import RepulsingHillClimbingOptimizer
from .simulated_annealing import SimulatedAnnealingOptimizer
from .spiral_optimization import SpiralOptimization
from .stochastic_hill_climbing import StochasticHillClimbingOptimizer
from .tree_structured_parzen_estimators import TreeStructuredParzenEstimators

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

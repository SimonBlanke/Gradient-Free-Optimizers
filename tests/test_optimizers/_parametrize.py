from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    PowellsMethod,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    PatternSearch,
    DirectAlgorithm,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    EvolutionStrategyOptimizer,
    LipschitzOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)

optimizers_singleOpt = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (RandomSearchOptimizer),
        (PowellsMethod),
        (PatternSearch),
        (DirectAlgorithm),
        (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (SimulatedAnnealingOptimizer),
    ],
)

optimizers_PopBased = (
    "Optimizer",
    [
        (DownhillSimplexOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (SpiralOptimization),
        (EvolutionStrategyOptimizer),
    ],
)

optimizers_noSBOM = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (RandomSearchOptimizer),
        (PowellsMethod),
        (PatternSearch),
        (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (SimulatedAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (SpiralOptimization),
        (EvolutionStrategyOptimizer),
        (DownhillSimplexOptimizer),
    ],
)

optimizers_SBOM = (
    "Optimizer",
    [
        (DirectAlgorithm),
        (LipschitzOptimizer),
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
    ],
)

optimizers_local = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
        (DownhillSimplexOptimizer),
    ],
)


optimizers = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        (DownhillSimplexOptimizer),
        (RandomSearchOptimizer),
        (PowellsMethod),
        (PatternSearch),
        (DirectAlgorithm),
        (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (SpiralOptimization),
        (EvolutionStrategyOptimizer),
        (LipschitzOptimizer),
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
    ],
)


optimizers_non_deterministic = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        # (DownhillSimplexOptimizer),
        (RandomSearchOptimizer),
        (PowellsMethod),
        (PatternSearch),
        # (GridSearchOptimizer),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (SpiralOptimization),
        (EvolutionStrategyOptimizer),
        (LipschitzOptimizer),
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
    ],
)

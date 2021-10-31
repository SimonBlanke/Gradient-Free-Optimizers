from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    PowellsMethod,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    PatternSearch,
    ParallelTemperingOptimizer,
    ParallelAnnealingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
    OneDimensionalBayesianOptimization,
    EnsembleOptimizer,
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
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (SimulatedAnnealingOptimizer),
        (DownhillSimplexOptimizer),
    ],
)

optimizers_PopBased = (
    "Optimizer",
    [
        (ParallelTemperingOptimizer),
        (ParallelAnnealingOptimizer),
        (ParticleSwarmOptimizer),
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
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (SimulatedAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParallelAnnealingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
        (DownhillSimplexOptimizer),
    ],
)

optimizers_SBOM = (
    "Optimizer",
    [
        (BayesianOptimizer),
        (OneDimensionalBayesianOptimization),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
        (EnsembleOptimizer),
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
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParallelAnnealingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
        (EnsembleOptimizer),
        (OneDimensionalBayesianOptimization),
    ],
)

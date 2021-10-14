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
    ParallelTemperingOptimizer,
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
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (SimulatedAnnealingOptimizer),
        # (DownhillSimplexOptimizer),
    ],
)

optimizers_PopBased = (
    "Optimizer",
    [
        (ParallelTemperingOptimizer),
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
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (SimulatedAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
        # (DownhillSimplexOptimizer),
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
        # (DownhillSimplexOptimizer),
    ],
)


optimizers = (
    "Optimizer",
    [
        (HillClimbingOptimizer),
        (StochasticHillClimbingOptimizer),
        (RepulsingHillClimbingOptimizer),
        (SimulatedAnnealingOptimizer),
        # (DownhillSimplexOptimizer),
        (RandomSearchOptimizer),
        (PowellsMethod),
        (RandomRestartHillClimbingOptimizer),
        (RandomAnnealingOptimizer),
        (ParallelTemperingOptimizer),
        (ParticleSwarmOptimizer),
        (EvolutionStrategyOptimizer),
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (ForestOptimizer),
        (EnsembleOptimizer),
        (OneDimensionalBayesianOptimization),
    ],
)

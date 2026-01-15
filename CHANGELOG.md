# Changelog

All notable changes to Gradient-Free-Optimizers are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For detailed release notes, see [GitHub Releases](https://github.com/SimonBlanke/Gradient-Free-Optimizers/releases).

## [Unreleased]

## [1.9.0] - 2026-01-15

Major release focusing on dependency reduction. scikit-learn and scipy are now optional dependencies with native Python implementations available for all core functionality.

### Added
- Private array backend (`_array_backend`) for pure Python array operations without NumPy
- Private math backend (`_math_backend`) for mathematical operations without SciPy
- Native `DecisionTreeRegressor` implementation
- Native `ExtraTreesRegressor` implementation
- Native `RandomForestRegressor` implementation
- Native `GradientBoostingRegressor` implementation
- `SimpleProgressBar` class as fallback when tqdm is unavailable
- Sigma self-adaptation for `EvolutionStrategyOptimizer`
- `convergence_threshold` parameter for Powell's Method
- Type hints to all optimizer classes and `Search` class
- Comprehensive docstrings for all optimizer classes
- Sphinx documentation with ReadTheDocs integration
- API tests for all optimizer categories

### Changed
- scikit-learn is now an optional dependency (native estimators used by default)
- SciPy is now an optional dependency
- tqdm is now an optional dependency
- Complete reimplementation of Powell's Method with improved line search algorithms
- Reworked README with new 3D optimization animation
- Consolidated CI workflows into single `ci.yml`
- Restructured test directory (`tests/test_main/`, `tests/test_internal/`, etc.)
- Improved error messages with actionable suggestions

### Removed
- `BayesianRidge` estimator
- Linear GP option from Gaussian Process regressor

### Fixed
- Golden section search algorithm in Powell's Method
- Mutable default argument anti-pattern (`constraints=[]` changed to `constraints=None`)
- Missing `@functools.wraps` on internal decorators
- Division by zero edge case in print-times
- Bug in evaluate method

## [1.8.1] - 2025-12-29

Re-release of v1.8.0 with updated package metadata.

## [1.8.0] - 2025-12-29

### Added
- Python 3.14 support
- Package keywords and classifiers for improved discoverability

### Changed
- Dropped Python 3.9 support
- Test performance improvements

### Fixed
- Sporadic test failures in CI

## [1.7.2] - 2025-09-21

### Added
- Native `GaussianProcessRegressor` implementation with RBF kernel
- Native `KernelDensityEstimator` implementation
- `Result` dataclass for structured evaluation results
- `ObjectiveAdapter` class for cleaner objective function handling
- Toy test functions (Sphere, Ackley) for benchmarking
- Python 3.13 support

### Changed
- Refactored memory/caching system using `CachedObjectiveAdapter`
- Refactored `ResultsManager` for improved result collection
- New optimization stopping implementation
- Performance improvements to `normalize` function (up to 90% faster)
- Performance improvements to `LipschitzFunction.find_best_slope` (81% faster)

### Fixed
- Issue with maximize/minimize objective function handling

## [1.7.1] - 2024-12-07

### Added
- Comprehensive docstrings for all optimizer classes

### Changed
- Dropped Python 3.8 support
- Improved type hints for `constraints`, `sampling`, and `initialize` parameters
- Refactored `move_climb` method to `CoreOptimizer`
- Cleaned up class inheritance and removed unused arguments

## [1.6.0] - 2024-08-14

### Added
- Python 3.12 support
- NumPy v2 and Pandas v2 compatibility
- PyTorch optimizer integration example

### Changed
- Migrated from `setup.py` to `pyproject.toml`
- Moved source code into `src/` directory structure

## [1.5.0] - 2024-07-22

### Added
- `GeneticAlgorithmOptimizer` for evolutionary optimization
- `DifferentialEvolutionOptimizer` for population-based optimization
- `mutation_rate` and `crossover_rate` parameters for evolutionary algorithms

### Changed
- Refactored stochastic hill climbing transition logic
- Moved discrete recombination method into base class

### Fixed
- Bug in constrained optimization

## [1.4.0] - 2024-05-11

### Added
- `OrthogonalGridSearchOptimizer` for systematic grid search
- `direction` parameter for grid search optimizer
- Search-space value validation

### Changed
- Dropped Python 3.5, 3.6, and 3.7 support
- Replaced `nth_iter` with `nth_trial` for clearer semantics
- Pandas v2 compatibility improvements

### Fixed
- Probability calculation bug in stochastic hill climbing
- Bugs in stochastic hill climbing and simulated annealing

## [1.3.0] - 2023-04-11

### Added
- Constrained optimization support for most optimizers
- Constrained optimization examples and documentation

### Changed
- Refactored optimizer and search classes into separate APIs

### Fixed
- Evaluation call from parent class

## [1.2.0] - 2022-10-20

### Added
- `SpiralOptimization` algorithm
- `LipschitzOptimizer` algorithm
- `DirectAlgorithm` (DIRECT) for global optimization
- Backend API for low-level optimizer control
- Python 3.10 and 3.11 support

### Changed
- Major refactoring for more consistent optimizer behavior
- Improved low-level API
- Refactored SMBO optimizers into unified pattern
- Refactored expected improvement into separate module
- Core optimizer moved into separate module

### Fixed
- Rotation matrix calculation
- Various fixes for DIRECT algorithm
- Grid search GCD calculation

## [1.0.1] - 2021-12-01

### Fixed
- Bug in grid search
- Random move and random state handling
- Various stability improvements from v1.0.0 release

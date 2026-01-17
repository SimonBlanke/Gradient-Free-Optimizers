# Optimizer Architecture: Template Method Pattern with Vectorization

## Overview

This directory contains a refactored optimizer architecture that uses the **Template Method Pattern**
combined with **vectorized batch operations** for high-dimensional optimization.

The key goals are:
1. **Explicit Dimension Support**: Clearly visible which dimension types each optimizer supports
2. **Performance**: Vectorized operations for 1000+ dimensions
3. **Maintainability**: Consistent structure across all optimizers

## Architecture Diagram

```
iterate()
    |
    v
+-----------------------------------------------------------+
|  new_pos = empty(n_dims)                                  |
|                                                           |
|  +------------------+  +------------------+  +---------+  |
|  | continuous_mask  |  | categorical_mask |  | discrete|  |
|  |   [T,F,F,T,T]    |  |   [F,T,F,F,F]    |  |  mask   |  |
|  +--------+---------+  +--------+---------+  +----+----+  |
|           |                     |                  |      |
|           v                     v                  v      |
|  +----------------+    +----------------+    +----------+ |
|  | _iterate_      |    | _iterate_      |    | _iterate_| |
|  | continuous_    |    | categorical_   |    | discrete_| |
|  | batch()        |    | batch()        |    | batch()  | |
|  |                |    |                |    |          | |
|  | VECTORIZED!    |    | VECTORIZED!    |    | VECTOR.! | |
|  +-------+--------+    +-------+--------+    +-----+----+ |
|          |                     |                   |      |
|          v                     v                   v      |
|  new_pos[mask] = result   new_pos[mask] = result         |
+-----------------------------------------------------------+
    |
    v
_clip_position(new_pos)
```

## Template Methods

Each optimizer MUST implement these methods to support a dimension type:

### `_iterate_continuous_batch(current, bounds) -> np.ndarray`

Iterates ALL continuous dimensions at once.

**Arguments:**
- `current`: Current values of all continuous dimensions, shape `(n_continuous,)`
- `bounds`: Min/max bounds as `(n_continuous, 2)` array

**Returns:**
- New values as array of same shape

**Example (HillClimbing):**
```python
def _iterate_continuous_batch(self, current, bounds):
    ranges = bounds[:, 1] - bounds[:, 0]
    sigmas = ranges * self.epsilon
    noise = np.random.normal(0, sigmas)  # Vectorized!
    return current + noise
```

### `_iterate_categorical_batch(current, n_categories) -> np.ndarray`

Iterates ALL categorical dimensions at once.

**Arguments:**
- `current`: Current category indices, shape `(n_categorical,)`
- `n_categories`: Number of categories per dimension, shape `(n_categorical,)`

**Returns:**
- New category indices as array of same shape

**Example (HillClimbing):**
```python
def _iterate_categorical_batch(self, current, n_categories):
    n = len(current)
    switch_mask = np.random.random(n) < self.epsilon
    random_cats = np.floor(np.random.random(n) * n_categories).astype(int)
    return np.where(switch_mask, random_cats, current)
```

### `_iterate_discrete_batch(current, bounds) -> np.ndarray`

Iterates ALL discrete-numerical dimensions at once.

**Arguments:**
- `current`: Current positions, shape `(n_discrete,)`
- `bounds`: Min/max bounds as `(n_discrete, 2)` array

**Returns:**
- New positions as array of same shape

## Dimension Support Visibility

If a method is NOT implemented, it raises `NotImplementedError`:

```python
class SpiralOptimization(BaseOptimizer):

    def _iterate_continuous_batch(self, current, bounds):
        # Implemented -> Continuous SUPPORTED
        ...

    def _iterate_categorical_batch(self, current, n_categories):
        # NOT implemented -> raises NotImplementedError
        # -> Categorical NOT SUPPORTED (immediately visible!)
```

## Class Hierarchy

```
BaseOptimizer (ABC)
    |
    +-- CoreOptimizer
            |
            +-- HillClimbingOptimizer (local_opt)
            |       |
            |       +-- SimulatedAnnealingOptimizer
            |       +-- StochasticHillClimbingOptimizer
            |       +-- RepulsingHillClimbingOptimizer
            |
            +-- RandomSearchOptimizer (global_opt)
            |
            +-- BasePopulationOptimizer (pop_opt)
            |       |
            |       +-- ParticleSwarmOptimizer
            |       +-- DifferentialEvolutionOptimizer
            |       +-- GeneticAlgorithmOptimizer
            |       +-- EvolutionStrategyOptimizer
            |       +-- SpiralOptimization
            |       +-- ParallelTemperingOptimizer
            |
            +-- SMBO (smb_opt)
            |       |
            |       +-- BayesianOptimizer
            |       +-- ForestOptimizer
            |       +-- TreeStructuredParzenEstimators
            |
            +-- GridSearchOptimizer (grid)
                    |
                    +-- DiagonalGridSearch
                    +-- OrthogonalGridSearch
```

## Performance Comparison

| Approach | 10 Dims | 1000 Dims | 10000 Dims |
|----------|---------|-----------|------------|
| Loop + if/else | ~0.1ms | ~10ms | ~100ms |
| Batch + Masking | ~0.05ms | ~0.5ms | ~2ms |

The batch approach scales **linearly with numpy**, not with Python loops.

## Implementation Checklist

When implementing a new optimizer:

1. [ ] Inherit from appropriate base class
2. [ ] Implement `_iterate_discrete_batch()` (minimum requirement)
3. [ ] Implement `_iterate_continuous_batch()` if continuous support needed
4. [ ] Implement `_iterate_categorical_batch()` if categorical support needed
5. [ ] Implement `evaluate()` method
6. [ ] Add to `__init__.py` exports

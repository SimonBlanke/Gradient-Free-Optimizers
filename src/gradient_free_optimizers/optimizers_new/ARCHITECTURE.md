# Optimizer Architecture: Template Method Pattern with Vectorization

## Overview

This directory contains a refactored optimizer architecture that uses the **Template Method Pattern**
combined with **vectorized batch operations** for high-dimensional optimization.

The key goals are:
1. **Explicit Dimension Support**: Clearly visible which dimension types each optimizer supports
2. **Performance**: Vectorized operations for 1000+ dimensions
3. **Maintainability**: Consistent structure across all optimizers
4. **Self-Contained**: No dependencies on the old `optimizers/` module

## Current Status

### Completed

| Component | Status | Tests |
|-----------|--------|-------|
| `CoreOptimizer` | DONE | - |
| `HillClimbingOptimizer` | DONE | 324 passed |
| `RandomSearchOptimizer` | DONE | 100 passed |
| `StochasticHillClimbingOptimizer` | DONE | 74 passed |
| `SimulatedAnnealingOptimizer` | DONE | 106 passed |
| `Converter` | DONE (copied) | - |
| `Initializer` | DONE (copied) | - |
| Search Integration | DONE | - |

### Pending Migration

| Optimizer | Priority | Notes |
|-----------|----------|-------|
| `RepulsingHillClimbingOptimizer` | High | Extends HillClimbing |
| `ParticleSwarmOptimizer` | Medium | Population-based |
| `DifferentialEvolutionOptimizer` | Medium | Population-based |
| `BayesianOptimizer` | Low | Complex surrogate model |

## Architecture

### Template Method Pattern: iterate()

```
iterate()                          <- CoreOptimizer (Orchestration)
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

### Template Method Pattern: evaluate()

```
evaluate(score_new)                <- CoreOptimizer (Orchestration)
    |
    +-> _track_score(score_new)    <- CoreOptimizer (Common)
    |       - scores_valid.append()
    |       - positions_valid.append()
    |       - nth_trial += 1
    |
    +-> _evaluate(score_new)       <- Subclass (Algorithm-specific)
            - HillClimbing: greedy + n_neighbours
            - SimulatedAnnealing: probabilistic acceptance
            - PSO: update personal best
            - DE: replace if better
```

## Position Semantics

Positions are stored differently based on dimension type:

| Dimension Type | Position Stores | Example |
|----------------|-----------------|---------|
| **Continuous** | Raw value | `0.05` (actual learning rate) |
| **Categorical** | Index | `1` (index into `["relu", "tanh"]`) |
| **Discrete** | Index | `2` (index into `np.array([1,2,4,8])`) |

**Why indices for Categorical/Discrete?**
- Non-uniform spacing (e.g., log-scale: `[0.001, 0.01, 0.1, 1.0]`)
- Noise/mutation operates in normalized index-space
- Consistent behavior regardless of value-spacing

The `Converter` handles translation between positions and user-facing values.

## Module Structure

```
optimizers_new/
|-- ARCHITECTURE.md              <- This file
|-- __init__.py                  <- Exports all optimizers
|-- base_optimizer.py            <- ABC with template method stubs
|
|-- core_optimizer/
|   |-- __init__.py
|   |-- core_optimizer.py        <- Orchestration + Search interface
|   |-- converter.py             <- Position <-> Value <-> Para
|   |-- init_positions.py        <- Initialization strategies
|   |-- _converter_memory.py     <- Memory/DataFrame operations
|   +-- utils.py                 <- set_random_seed, move_random
|
|-- local_opt/
|   |-- __init__.py
|   |-- hill_climbing_optimizer.py    <- IMPLEMENTED
|   |-- simulated_annealing.py        <- Skeleton
|   |-- stochastic_hill_climbing.py   <- Skeleton
|   +-- ...
|
|-- global_opt/                  <- Skeletons
|-- pop_opt/                     <- Skeletons
|-- smb_opt/                     <- Skeletons
+-- grid/                        <- Skeletons
```

## Search Integration

The `CoreOptimizer` provides all methods required by the `Search` class:

```python
class CoreOptimizer(BaseOptimizer):
    # Required by Search
    self.conv                    # Converter instance
    self.init                    # Initializer instance
    self.optimizers = [self]     # List of optimizers (for tracking)

    def init_pos(self):          # Get next init position
    def evaluate_init(score):    # Handle init phase
    def finish_initialization(): # Transition to iteration
    def iterate():               # Generate new position
    def evaluate(score):         # Handle iteration score

    # Properties with setters (Search sets these)
    best_score                   # Best score found
    best_value                   # Best values (raw)
    best_para                    # Best parameters (dict)
```

## How to Migrate an Optimizer

### Step 1: Implement the Optimizer

Create the optimizer in the appropriate subdirectory:

```python
# optimizers_new/local_opt/simulated_annealing.py

from ..core_optimizer import CoreOptimizer

class SimulatedAnnealingOptimizer(CoreOptimizer):
    def __init__(self, search_space, ..., start_temp=10.0, annealing_rate=0.98):
        super().__init__(search_space, ...)
        self.temp = start_temp
        self.annealing_rate = annealing_rate

    def _iterate_continuous_batch(self, current, bounds):
        # Same as HillClimbing (can inherit or copy)
        ...

    def _iterate_categorical_batch(self, current, n_categories):
        ...

    def _iterate_discrete_batch(self, current, bounds):
        ...

    def _evaluate(self, score_new):
        # Metropolis criterion: accept worse with probability
        if score_new > self.score_current:
            self._update_current(self.pos_new, score_new)
        else:
            delta = self.score_current - score_new
            accept_prob = np.exp(-delta / self.temp)
            if np.random.random() < accept_prob:
                self._update_current(self.pos_new, score_new)

        self._update_best(self.pos_new, score_new)
        self.temp *= self.annealing_rate  # Cool down
```

### Step 2: Export in __init__.py

```python
# optimizers_new/local_opt/__init__.py
from .simulated_annealing import SimulatedAnnealingOptimizer
```

### Step 3: Swap Import in optimizer_search/

```python
# optimizer_search/simulated_annealing.py

# OLD:
# from ..optimizers import SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer

# NEW:
from ..optimizers_new import SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer
```

### Step 4: Run Tests

```bash
pytest tests/ -k "SimulatedAnnealing" -v
```

All existing tests should pass without modification.

## Template Methods Reference

### _iterate_continuous_batch(current, bounds) -> np.ndarray

Iterates ALL continuous dimensions at once (vectorized).

**Arguments:**
- `current`: Current values, shape `(n_continuous,)`
- `bounds`: Min/max bounds, shape `(n_continuous, 2)`

**Returns:** New values, shape `(n_continuous,)`

### _iterate_categorical_batch(current, n_categories) -> np.ndarray

Iterates ALL categorical dimensions at once (vectorized).

**Arguments:**
- `current`: Current category indices, shape `(n_categorical,)`
- `n_categories`: Number of categories per dimension, shape `(n_categorical,)`

**Returns:** New category indices, shape `(n_categorical,)`

### _iterate_discrete_batch(current, bounds) -> np.ndarray

Iterates ALL discrete-numerical dimensions at once (vectorized).

**Arguments:**
- `current`: Current positions (indices), shape `(n_discrete,)`
- `bounds`: Min/max bounds, shape `(n_discrete, 2)`

**Returns:** New positions, shape `(n_discrete,)`

### _evaluate(score_new) -> None

Algorithm-specific evaluation logic. Called after `_track_score()`.

**Arguments:**
- `score_new`: Score of the position in `self.pos_new`

**Should:**
- Implement acceptance criteria (greedy, probabilistic, etc.)
- Call `_update_current()` and `_update_best()` as appropriate

## Performance

| Approach | 10 Dims | 1000 Dims | 10000 Dims |
|----------|---------|-----------|------------|
| Loop + if/else | ~0.1ms | ~10ms | ~100ms |
| Batch + Masking | ~0.05ms | ~0.5ms | ~2ms |

The batch approach scales **linearly with numpy**, not with Python loops.

## Implementation Checklist

When implementing a new optimizer:

- [ ] Inherit from `CoreOptimizer`
- [ ] Implement `_iterate_continuous_batch()` if continuous support needed
- [ ] Implement `_iterate_categorical_batch()` if categorical support needed
- [ ] Implement `_iterate_discrete_batch()` for discrete support
- [ ] Implement `_evaluate()` with algorithm-specific logic
- [ ] Add to `local_opt/__init__.py` (or appropriate submodule)
- [ ] Add to `optimizers_new/__init__.py`
- [ ] Swap import in `optimizer_search/`
- [ ] Run tests: `pytest tests/ -k "OptimizerName" -v`
- [ ] Add optimizer to `tests/test_dimension_types.py` OPTIMIZERS list
- [ ] Add optimizer to `tests/test_vectorization.py` OPTIMIZERS list

## Testing Strategy

We have two dedicated test files for verifying optimizer implementations:

### test_dimension_types.py

Tests that each optimizer correctly handles all dimension types:

| Test Category | What it Tests |
|---------------|---------------|
| Continuous | Values in range, float types, single dimension |
| Categorical | Valid options, preserved types (str, bool), binary categories |
| Discrete | Exact value match (no interpolation), valid values |
| Mixed | All three types combined, correct types per dimension |
| Edge Cases | Narrow ranges, large category sets, single dimensions |
| Reproducibility | Same random_state = same results |

**Usage:**
```bash
# Run all dimension type tests
pytest optimizers_new/tests/test_dimension_types.py -v

# Run for specific optimizer
pytest optimizers_new/tests/test_dimension_types.py -v -k "HillClimbing"
```

### test_vectorization.py

Tests that batch operations work correctly with many dimensions:

| Test Category | What it Tests |
|---------------|---------------|
| Mask Creation | Correct masks for continuous/categorical/discrete |
| Bounds Arrays | Correct shapes for bounds/sizes arrays |
| Batch Methods | Methods receive vectorized inputs |
| 50 Dimensions | Moderate scale, fast for CI |
| 200+ Dimensions | Marked `@pytest.mark.slow` for optional runs |
| Position Arrays | Correct length and value types |
| Clipping | All values stay within valid bounds |

**CI Strategy:**
- Default tests use 50 dimensions (fast, ~1 second)
- High-dimension tests (200+) are marked `@pytest.mark.slow`
- Run slow tests periodically: `pytest -m slow`

**Usage:**
```bash
# Run fast vectorization tests only
pytest optimizers_new/tests/test_vectorization.py -v -m "not slow"

# Run all tests including slow ones
pytest optimizers_new/tests/test_vectorization.py -v
```

### Adding a New Optimizer to Tests

When implementing a new optimizer, add it to both test files:

```python
# In test_dimension_types.py AND test_vectorization.py

from ..local_opt import HillClimbingOptimizer
from ..global_opt import RandomSearchOptimizer
from ..local_opt import YourNewOptimizer  # Add import

OPTIMIZERS = [
    HillClimbingOptimizer,
    RandomSearchOptimizer,
    YourNewOptimizer,  # Add to list
]
```

All parametrized tests will automatically run for the new optimizer.

## Key Fixes During Integration

When integrating with the `Search` class, we discovered several requirements:

1. **Cooperative Multiple Inheritance**: `BaseOptimizer.__init__()` must call `super().__init__()` to ensure `Search.__init__()` runs.

2. **Property Setters**: `Search` sets `best_score`, `best_value`, `best_para` directly, so these need setters.

3. **Optimizer List**: `self.optimizers = [self]` is required for search tracking tests.

4. **No search() in BaseOptimizer**: The `Search` class provides `search()`, so `BaseOptimizer` should not define it.

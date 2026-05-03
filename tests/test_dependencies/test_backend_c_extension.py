"""Backend isolation tests for the C extension tier.

Verifies that Gradient-Free-Optimizers works correctly with the
C extension backend (no numpy, no scipy). The C extension accelerates
arithmetic via compiled C functions while falling back to GFOArray
for everything else.

Three test layers:
  1. Environment verification (C extension active, numpy absent)
  2. Unit tests (C-accelerated operations produce correct results)
  3. Integration tests (optimizers run and use _CGFOArray)
"""

import pytest

try:
    import numpy

    pytest.skip(
        "numpy is installed - test requires numpy to be absent",
        allow_module_level=True,
    )
except ImportError:
    pass

from gradient_free_optimizers._array_backend import (
    HAS_C_EXTENSION,
    HAS_NUMPY,
    _backend_name,
    array,
    clip,
    exp,
    linspace,
    sqrt,
    zeros,
)
from gradient_free_optimizers._array_backend._pure import GFOArray

if not HAS_C_EXTENSION:
    pytest.skip(
        "C extension not available - test requires compiled _fast_ops",
        allow_module_level=True,
    )

from gradient_free_optimizers._array_backend._c_extension import _CGFOArray


class TestEnvironmentVerification:
    """Layer 1: Verify that C extension is active and numpy is absent."""

    def test_numpy_not_importable(self):
        with pytest.raises(ImportError):
            import numpy  # noqa: F401

    def test_scipy_not_importable(self):
        with pytest.raises(ImportError):
            import scipy  # noqa: F401

    def test_c_extension_available(self):
        assert HAS_C_EXTENSION, "C extension should be compiled and loadable"

    def test_numpy_flag_false(self):
        assert not HAS_NUMPY

    def test_backend_name_is_c_extension(self):
        assert (
            _backend_name == "c_extension"
        ), f"Expected 'c_extension' backend, got '{_backend_name}'"

    def test_array_type_is_cgfoarray(self):
        a = array([1.0, 2.0])
        assert type(a) is _CGFOArray, f"Expected _CGFOArray, got {type(a).__name__}"

    def test_cgfoarray_is_gfoarray_subclass(self):
        assert issubclass(_CGFOArray, GFOArray)

    def test_linspace_returns_gfoarray(self):
        a = linspace(0, 10, 5)
        assert isinstance(a, GFOArray)

    def test_arithmetic_produces_cgfoarray(self):
        """C-accelerated arithmetic promotes GFOArray to _CGFOArray."""
        a = array([1.0, 2.0])
        b = array([3.0, 4.0])
        c = a + b
        assert type(c) is _CGFOArray


class TestCAcceleratedOps:
    """Layer 2: Verify that C-accelerated code paths are taken.

    Each test checks both correctness AND that the result type
    is _CGFOArray (proving the C path was taken, not the fallback).
    """

    def test_add_same_size(self):
        a = array([1.0, 2.0, 3.0])
        b = array([4.0, 5.0, 6.0])
        c = a + b
        assert list(c._data) == [5.0, 7.0, 9.0]
        assert type(c) is _CGFOArray, "C path should return _CGFOArray"

    def test_add_scalar(self):
        a = array([1.0, 2.0, 3.0])
        c = a + 10.0
        assert list(c._data) == [11.0, 12.0, 13.0]
        assert type(c) is _CGFOArray

    def test_add_2d_broadcast_falls_to_pure(self):
        """2D + 1D falls back to pure Python broadcasting."""
        a = array([[1.0, 2.0], [3.0, 4.0]])
        b = array([10.0, 20.0])
        c = a + b
        assert c._shape == (2, 2)
        assert list(c._data) == [11.0, 22.0, 13.0, 24.0]

    def test_sub_same_size(self):
        c = array([10.0, 20.0]) - array([1.0, 2.0])
        assert list(c._data) == [9.0, 18.0]
        assert type(c) is _CGFOArray

    def test_mul_same_size(self):
        c = array([2.0, 3.0]) * array([4.0, 5.0])
        assert list(c._data) == [8.0, 15.0]
        assert type(c) is _CGFOArray

    def test_mul_scalar(self):
        c = array([2.0, 3.0]) * 10.0
        assert list(c._data) == [20.0, 30.0]
        assert type(c) is _CGFOArray

    def test_neg(self):
        c = -array([1.0, -2.0])
        assert list(c._data) == [-1.0, 2.0]
        assert type(c) is _CGFOArray

    def test_sum(self):
        from gradient_free_optimizers._array_backend import sum as arr_sum

        assert arr_sum(array([1.0, 2.0, 3.0])) == 6.0

    def test_argmax(self):
        from gradient_free_optimizers._array_backend import argmax

        assert argmax(array([1.0, 5.0, 3.0])) == 1

    def test_dot(self):
        from gradient_free_optimizers._array_backend import dot

        assert dot(array([1.0, 2.0, 3.0]), array([4.0, 5.0, 6.0])) == 32.0

    def test_matmul_2d(self):
        a = array([[1.0, 0.0], [0.0, 1.0]])
        b = array([[3.0, 4.0], [5.0, 6.0]])
        c = a @ b
        assert c._shape == (2, 2)
        assert list(c._data) == [3.0, 4.0, 5.0, 6.0]
        assert type(c) is _CGFOArray

    def test_exp(self):
        r = exp(array([0.0]))
        assert abs(r._data[0] - 1.0) < 1e-10
        assert type(r) is _CGFOArray

    def test_sqrt(self):
        r = sqrt(array([4.0, 9.0]))
        assert list(r._data) == [2.0, 3.0]
        assert type(r) is _CGFOArray

    def test_clip(self):
        r = clip(array([-1.0, 5.0, 15.0]), 0.0, 10.0)
        assert list(r._data) == [0.0, 5.0, 10.0]
        assert type(r) is _CGFOArray

    def test_type_preserved_through_chain(self):
        """Verify C type is preserved across multiple operations."""
        a = array([1.0, 2.0, 3.0])
        b = array([4.0, 5.0, 6.0])
        r = (a + b) * 2.0 - a
        assert (
            type(r) is _CGFOArray
        ), "Type should stay _CGFOArray through add -> mul -> sub chain"


class TestFallbackOps:
    """Layer 2: Operations not in C fall back to pure Python correctly."""

    def test_argsort(self):
        a = array([3.0, 1.0, 2.0])
        idx = a.argsort()
        assert list(idx._data) == [1, 2, 0]

    def test_min_axis(self):
        a = array([[3.0, 1.0], [4.0, 2.0]])
        r = a.min(1)
        assert list(r._data) == [1.0, 2.0]

    def test_and_boolean(self):
        a = array([1.0, 0.0]) != 0
        b = array([1.0, 1.0]) != 0
        c = a & b
        assert c._data == [True, False]

    def test_fancy_index_float_coercion(self):
        a = array([10.0, 20.0, 30.0])
        idx = array([0.0, 2.0]).astype(int)
        r = a[idx]
        assert list(r._data) == [10.0, 30.0]

    def test_normal_array_scale(self):
        from gradient_free_optimizers._array_backend import random

        rng = random.default_rng(42)
        scales = array([0.1, 0.5, 1.0])
        r = rng.normal(0.0, scales)
        assert len(r._data) == 3

    def test_integers_array_bounds(self):
        from gradient_free_optimizers._array_backend import random

        rng = random.default_rng(42)
        lows = array([0.0, 0.0])
        highs = array([10.0, 20.0])
        r = rng.integers(lows, highs)
        assert len(r._data) == 2


ALL_OPTIMIZERS = []

from gradient_free_optimizers import (  # noqa: E402
    BayesianOptimizer,
    DifferentialEvolutionOptimizer,
    DirectAlgorithm,
    DownhillSimplexOptimizer,
    EvolutionStrategyOptimizer,
    ForestOptimizer,
    GeneticAlgorithmOptimizer,
    GridSearchOptimizer,
    HillClimbingOptimizer,
    LipschitzOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomAnnealingOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    SpiralOptimization,
    StochasticHillClimbingOptimizer,
    TreeStructuredParzenEstimators,
)

ALL_OPTIMIZERS = [
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
]

SEARCH_SPACE = {
    "x1": linspace(-5, 5, 11),
    "x2": linspace(-5, 5, 11),
}


def objective_function(para):
    return -(para["x1"] ** 2 + para["x2"] ** 2)


class TestIntegrationCExtension:
    """Layer 3: Optimizer integration tests verifying C extension usage."""

    @pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
    def test_optimizer_runs(self, Optimizer):
        opt = Optimizer(
            SEARCH_SPACE,
            initialize={"random": 3},
            random_state=42,
        )
        opt.search(
            objective_function,
            n_iter=10,
            verbosity=False,
        )

        assert opt.best_para is not None
        assert opt.best_score is not None

    def test_c_extension_used_in_arithmetic(self):
        """Verify C path is taken and type is preserved through chains."""
        a = array([1.0, 2.0, 3.0])
        b = array([4.0, 5.0, 6.0])
        c = a + b
        assert (
            type(c) is _CGFOArray
        ), f"First add returned {type(c).__name__}, expected _CGFOArray"
        d = c * 2.0
        assert (
            type(d) is _CGFOArray
        ), f"Chained mul returned {type(d).__name__}, expected _CGFOArray"
        e = d - a
        assert (
            type(e) is _CGFOArray
        ), f"Chained sub returned {type(e).__name__}, expected _CGFOArray"

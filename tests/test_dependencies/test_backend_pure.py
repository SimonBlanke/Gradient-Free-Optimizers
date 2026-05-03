"""Backend isolation tests for the pure Python tier.

Verifies that Gradient-Free-Optimizers works correctly when
numpy, scipy, and the C extension are all absent. This is the
lowest-performance tier: only stdlib + tqdm + pandas.

Three test layers:
  1. Environment verification (correct backend is active)
  2. Unit tests (backend functions produce correct results)
  3. Integration tests (optimizers run and use the pure backend)
"""

import os

import pytest

_ci_strict = os.environ.get("GFO_CI_STRICT")

try:
    import numpy

    if _ci_strict:
        raise RuntimeError(
            "numpy is installed but must be absent (GFO_CI_STRICT is set)"
        )
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
    argsort,
    array,
    clip,
    linspace,
    maximum,
    minimum,
    ones,
    zeros,
)
from gradient_free_optimizers._array_backend._pure import GFOArray
from gradient_free_optimizers._math_backend import (
    HAS_SCIPY,
    cdist,
    norm_cdf,
    norm_pdf,
)


class TestEnvironmentVerification:
    """Layer 1: Verify that we are running in a pure Python environment."""

    def test_numpy_not_importable(self):
        with pytest.raises(ImportError):
            import numpy  # noqa: F401

    def test_scipy_not_importable(self):
        with pytest.raises(ImportError):
            import scipy  # noqa: F401

    def test_c_extension_not_available(self):
        assert (
            not HAS_C_EXTENSION
        ), "C extension should not be available in pure Python tier"

    def test_numpy_flag_false(self):
        assert not HAS_NUMPY

    def test_backend_name_is_pure(self):
        assert (
            _backend_name == "pure"
        ), f"Expected 'pure' backend, got '{_backend_name}'"

    def test_array_type_is_gfoarray(self):
        a = array([1.0, 2.0])
        assert type(a) is GFOArray, f"Expected GFOArray, got {type(a).__name__}"

    def test_linspace_returns_gfoarray(self):
        a = linspace(0, 10, 5)
        assert type(a) is GFOArray


class TestBackendUnitOps:
    """Layer 2: Unit tests for pure Python array operations."""

    def test_array_creation(self):
        a = array([1.0, 2.0, 3.0])
        assert list(a._data) == [1.0, 2.0, 3.0]
        assert a._shape == (3,)

    def test_array_2d(self):
        a = array([[1.0, 2.0], [3.0, 4.0]])
        assert a._shape == (2, 2)
        assert a._ndim == 2

    def test_add_same_size(self):
        a = array([1.0, 2.0])
        b = array([3.0, 4.0])
        c = a + b
        assert list(c._data) == [4.0, 6.0]
        assert type(c) is GFOArray

    def test_add_scalar(self):
        a = array([1.0, 2.0]) + 10.0
        assert list(a._data) == [11.0, 12.0]

    def test_add_2d_broadcast_1d(self):
        a = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = array([10.0, 20.0, 30.0])
        c = a + b
        assert c._shape == (2, 3)
        assert list(c._data) == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]

    def test_mul_same_size(self):
        a = array([2.0, 3.0])
        b = array([4.0, 5.0])
        c = a * b
        assert list(c._data) == [8.0, 15.0]
        assert type(c) is GFOArray

    def test_sub(self):
        c = array([10.0, 20.0]) - array([1.0, 2.0])
        assert list(c._data) == [9.0, 18.0]

    def test_clip(self):
        a = clip(array([-1.0, 5.0, 15.0]), 0.0, 10.0)
        assert list(a._data) == [0.0, 5.0, 10.0]

    def test_zeros(self):
        a = zeros(3)
        assert list(a._data) == [0.0, 0.0, 0.0]
        assert type(a) is GFOArray

    def test_ones(self):
        a = ones(3)
        assert list(a._data) == [1.0, 1.0, 1.0]

    def test_maximum(self):
        a = maximum(array([1.0, 5.0]), array([3.0, 2.0]))
        assert list(a._data) == [3.0, 5.0]

    def test_minimum(self):
        a = minimum(array([1.0, 5.0]), array([3.0, 2.0]))
        assert list(a._data) == [1.0, 2.0]

    def test_argsort(self):
        a = array([3.0, 1.0, 2.0])
        idx = a.argsort()
        assert list(idx._data) == [1, 2, 0]

    def test_argsort_module_level(self):
        idx = argsort(array([3.0, 1.0, 2.0]))
        assert list(idx._data) == [1, 2, 0]

    def test_min_axis1(self):
        a = array([[3.0, 1.0], [4.0, 2.0]])
        r = a.min(1)
        assert list(r._data) == [1.0, 2.0]

    def test_max_axis0(self):
        a = array([[3.0, 1.0], [4.0, 2.0]])
        r = a.max(0)
        assert list(r._data) == [4.0, 2.0]

    def test_and_boolean(self):
        a = array([1.0, 0.0]) != 0
        b = array([0.0, 0.0]) != 0
        c = a & b
        assert c._data == [False, False]

    def test_or_boolean(self):
        a = array([1.0, 0.0]) != 0
        b = array([0.0, 1.0]) != 0
        c = a | b
        assert c._data == [True, True]

    def test_fancy_index_float_coercion(self):
        a = array([10.0, 20.0, 30.0])
        idx = array([0.0, 2.0]).astype(int)
        r = a[idx]
        assert list(r._data) == [10.0, 30.0]


class TestRNGOps:
    """Layer 2: Unit tests for pure Python RNG with array args."""

    def test_normal_scalar_scale(self):
        from gradient_free_optimizers._array_backend import random

        rng = random.default_rng(42)
        r = rng.normal(0.0, 1.0, 5)
        assert isinstance(r, GFOArray)
        assert len(r._data) == 5

    def test_normal_array_scale(self):
        from gradient_free_optimizers._array_backend import random

        rng = random.default_rng(42)
        scales = array([0.1, 0.5, 1.0])
        r = rng.normal(0.0, scales)
        assert isinstance(r, GFOArray)
        assert len(r._data) == 3

    def test_integers_scalar(self):
        from gradient_free_optimizers._array_backend import random

        rng = random.default_rng(42)
        r = rng.integers(0, 10, size=5)
        assert isinstance(r, GFOArray)
        assert len(r._data) == 5

    def test_integers_array_bounds(self):
        from gradient_free_optimizers._array_backend import random

        rng = random.default_rng(42)
        lows = array([0.0, 0.0, 0.0])
        highs = array([10.0, 20.0, 30.0])
        r = rng.integers(lows, highs)
        assert isinstance(r, GFOArray)
        assert len(r._data) == 3


class TestMathBackend:
    """Layer 2: Unit tests for pure Python math functions."""

    def test_norm_cdf_scalar(self):
        r = norm_cdf(0.0)
        assert abs(r - 0.5) < 1e-6

    def test_norm_cdf_1d(self):
        r = norm_cdf(array([0.0, 0.0]))
        assert len(r._data) == 2
        assert abs(r._data[0] - 0.5) < 1e-6

    def test_norm_cdf_2d(self):
        r = norm_cdf(array([[0.0], [0.0]]))
        assert r._shape == (2, 1)
        assert abs(r._data[0] - 0.5) < 1e-6

    def test_norm_pdf_scalar(self):
        r = norm_pdf(0.0)
        assert abs(r - 0.3989422804) < 1e-6

    def test_norm_pdf_2d(self):
        r = norm_pdf(array([[0.0], [0.0]]))
        assert r._shape == (2, 1)

    def test_cdist(self):
        a = array([[0.0, 0.0], [1.0, 1.0]])
        b = array([[1.0, 0.0]])
        r = cdist(a, b)
        assert r._shape == (2, 1)
        assert abs(r._data[0] - 1.0) < 1e-10

    def test_math_backend_is_pure(self):
        assert not HAS_SCIPY


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


class TestIntegrationPure:
    """Layer 3: Optimizer integration tests verifying pure backend usage."""

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

    @pytest.mark.parametrize("Optimizer", ALL_OPTIMIZERS)
    def test_optimizer_uses_pure_backend(self, Optimizer):
        """Verify that optimizer arrays are GFOArray, not _CGFOArray."""
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

        space_val = list(opt.conv.search_space.values())[0]
        assert type(space_val) is GFOArray, (
            f"Search space array is {type(space_val).__name__}, "
            f"expected GFOArray (pure Python)"
        )

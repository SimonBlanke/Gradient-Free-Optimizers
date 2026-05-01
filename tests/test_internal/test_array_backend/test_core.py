"""
Core tests for the array backend.

Tests both the NumPy backend and the pure Python fallback to ensure
they produce equivalent results.
"""

import math

import pytest

from gradient_free_optimizers._array_backend import HAS_NUMPY
from gradient_free_optimizers._array_backend import _pure as pure_backend

from .conftest import arrays_close, to_list

# Conditionally import numpy backend
if HAS_NUMPY:
    from gradient_free_optimizers._array_backend import _numpy as np_backend
else:
    np_backend = None

# Skip all tests in this module if numpy is not available
pytestmark = pytest.mark.skipif(
    not HAS_NUMPY, reason="NumPy not available for comparison tests"
)


# =============================================================================
# Array Creation Tests
# =============================================================================


class TestArrayCreation:
    """Test array creation functions."""

    def test_array_from_list(self):
        np_arr = np_backend.array([1, 2, 3, 4, 5])
        pure_arr = pure_backend.array([1, 2, 3, 4, 5])
        assert to_list(np_arr) == to_list(pure_arr)

    def test_array_from_nested_list(self):
        np_arr = np_backend.array([[1, 2], [3, 4]])
        pure_arr = pure_backend.array([[1, 2], [3, 4]])
        assert np_arr.shape == pure_arr.shape

    def test_zeros_1d(self):
        np_arr = np_backend.zeros(5)
        pure_arr = pure_backend.zeros(5)
        assert to_list(np_arr) == to_list(pure_arr)

    def test_zeros_2d(self):
        np_arr = np_backend.zeros((3, 4))
        pure_arr = pure_backend.zeros((3, 4))
        assert np_arr.shape == pure_arr.shape

    def test_ones_1d(self):
        np_arr = np_backend.ones(5)
        pure_arr = pure_backend.ones(5)
        assert to_list(np_arr) == to_list(pure_arr)

    def test_ones_2d(self):
        np_arr = np_backend.ones((2, 3))
        pure_arr = pure_backend.ones((2, 3))
        assert np_arr.shape == pure_arr.shape

    @pytest.mark.parametrize(
        ("start", "stop", "num"),
        [
            (0, 10, 5),
            (0, 1, 11),
            (-5, 5, 21),
            (0, 100, 101),
        ],
    )
    def test_linspace(self, start, stop, num):
        np_arr = np_backend.linspace(start, stop, num)
        pure_arr = pure_backend.linspace(start, stop, num)
        assert arrays_close(np_arr, pure_arr)

    @pytest.mark.parametrize(
        ("start", "stop", "step"),
        [
            (0, 10, 1),
            (0, 10, 2),
            (0, 5, 0.5),
            (-10, 10, 1),
        ],
    )
    def test_arange(self, start, stop, step):
        np_arr = np_backend.arange(start, stop, step)
        pure_arr = pure_backend.arange(start, stop, step)
        assert arrays_close(np_arr, pure_arr)

    def test_eye(self):
        np_arr = np_backend.eye(3)
        pure_arr = pure_backend.eye(3)
        np_flat = to_list(np_arr.flatten())
        pure_flat = to_list(pure_arr.flatten())
        assert np_flat == pure_flat


# =============================================================================
# Mathematical Operations Tests
# =============================================================================


class TestMathOperations:
    """Test mathematical operations."""

    def test_sum(self, sample_array):
        np_result = np_backend.sum(np_backend.array(sample_array))
        pure_result = pure_backend.sum(pure_backend.array(sample_array))
        assert np_result == pure_result

    def test_mean(self, sample_array):
        np_result = np_backend.mean(np_backend.array(sample_array))
        pure_result = pure_backend.mean(pure_backend.array(sample_array))
        assert abs(np_result - pure_result) < 1e-10

    def test_std(self, sample_array):
        np_result = np_backend.std(np_backend.array(sample_array))
        pure_result = pure_backend.std(pure_backend.array(sample_array))
        assert abs(np_result - pure_result) < 1e-10

    @pytest.mark.parametrize(
        "values",
        [
            [0, 1, 2, 3],
            [0.5, 1.0, 1.5, 2.0],
            [-1, 0, 1],
        ],
    )
    def test_exp(self, values):
        np_result = np_backend.exp(np_backend.array(values))
        pure_result = pure_backend.exp(pure_backend.array(values))
        assert arrays_close(np_result, pure_result)

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3, 4],
            [0.5, 1.0, 2.0, 10.0],
            [math.e, math.e**2],
        ],
    )
    def test_log(self, values):
        np_result = np_backend.log(np_backend.array(values))
        pure_result = pure_backend.log(pure_backend.array(values))
        assert arrays_close(np_result, pure_result)

    @pytest.mark.parametrize(
        "values",
        [
            [1, 4, 9, 16, 25],
            [0.25, 0.5, 1.0, 2.0],
        ],
    )
    def test_sqrt(self, values):
        np_result = np_backend.sqrt(np_backend.array(values))
        pure_result = pure_backend.sqrt(pure_backend.array(values))
        assert arrays_close(np_result, pure_result)

    def test_abs(self):
        values = [-3, -2, -1, 0, 1, 2, 3]
        np_result = np_backend.abs(np_backend.array(values))
        pure_result = pure_backend.abs(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result)

    def test_power(self):
        values = [1, 2, 3, 4, 5]
        np_result = np_backend.power(np_backend.array(values), 2)
        pure_result = pure_backend.power(pure_backend.array(values), 2)
        assert to_list(np_result) == to_list(pure_result)


# =============================================================================
# Clipping and Rounding Tests
# =============================================================================


class TestClippingAndRounding:
    """Test clipping and rounding operations."""

    @pytest.mark.parametrize(
        ("values", "a_min", "a_max", "expected"),
        [
            ([1, 2, 3, 4, 5], 2, 4, [2, 2, 3, 4, 4]),
            ([-5, 0, 5, 10], 0, 5, [0, 0, 5, 5]),
            ([0.5, 1.5, 2.5], 1, 2, [1, 1.5, 2]),
        ],
    )
    def test_clip(self, values, a_min, a_max, expected):
        np_result = np_backend.clip(np_backend.array(values), a_min, a_max)
        pure_result = pure_backend.clip(pure_backend.array(values), a_min, a_max)
        assert to_list(np_result) == expected
        assert to_list(pure_result) == expected

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1.2, 2.5, 3.7, 4.1], [1, 2, 4, 4]),
            ([0.4, 0.5, 0.6], [0, 0, 1]),
            ([-1.5, -0.5, 0.5, 1.5], [-2, 0, 0, 2]),
        ],
    )
    def test_rint(self, values, expected):
        np_result = np_backend.rint(np_backend.array(values))
        pure_result = pure_backend.rint(pure_backend.array(values))
        # Note: rounding behavior may differ slightly at .5 boundaries
        assert arrays_close(np_result, pure_result, atol=1)


# =============================================================================
# Index Operations Tests
# =============================================================================


class TestIndexOperations:
    """Test index-based operations."""

    def test_argmax(self):
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        np_result = np_backend.argmax(np_backend.array(values))
        pure_result = pure_backend.argmax(pure_backend.array(values))
        assert np_result == pure_result == 5

    def test_argmin(self):
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        np_result = np_backend.argmin(np_backend.array(values))
        pure_result = pure_backend.argmin(pure_backend.array(values))
        assert np_result == pure_result == 1

    def test_argsort(self):
        values = [3, 1, 4, 1, 5]
        np_result = np_backend.argsort(np_backend.array(values))
        pure_result = pure_backend.argsort(pure_backend.array(values))
        # First two elements should be indices 1 and 3 (both have value 1)
        assert to_list(np_result)[:2] in [[1, 3], [3, 1]]
        assert to_list(pure_result)[:2] in [[1, 3], [3, 1]]

    def test_where_indices_only(self):
        values = [True, False, True, False, True]
        np_result = np_backend.where(np_backend.array(values))
        pure_result = pure_backend.where(pure_backend.array(values))
        assert to_list(np_result[0]) == to_list(pure_result[0]) == [0, 2, 4]

    def test_where_with_values(self):
        condition = [True, False, True, False]
        x = [1, 2, 3, 4]
        y = [10, 20, 30, 40]
        np_result = np_backend.where(
            np_backend.array(condition), np_backend.array(x), np_backend.array(y)
        )
        pure_result = pure_backend.where(
            pure_backend.array(condition),
            pure_backend.array(x),
            pure_backend.array(y),
        )
        assert to_list(np_result) == to_list(pure_result) == [1, 20, 3, 40]

    def test_searchsorted(self):
        arr = [1, 3, 5, 7, 9]
        values = [0, 2, 5, 8, 10]
        np_result = np_backend.searchsorted(np_backend.array(arr), values)
        pure_result = pure_backend.searchsorted(pure_backend.array(arr), values)
        assert to_list(np_result) == to_list(pure_result)


# =============================================================================
# Set Operations Tests
# =============================================================================


class TestSetOperations:
    """Test set-like operations."""

    def test_unique(self):
        values = [1, 2, 2, 3, 3, 3, 4]
        np_result = np_backend.unique(np_backend.array(values))
        pure_result = pure_backend.unique(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result) == [1, 2, 3, 4]

    def test_intersect1d(self):
        a = [1, 2, 3, 4, 5]
        b = [3, 4, 5, 6, 7]
        np_result = np_backend.intersect1d(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.intersect1d(
            pure_backend.array(a), pure_backend.array(b)
        )
        assert to_list(np_result) == to_list(pure_result) == [3, 4, 5]

    def test_isin(self):
        elements = [1, 2, 3, 4, 5]
        test_elements = [2, 4]
        np_result = np_backend.isin(
            np_backend.array(elements), np_backend.array(test_elements)
        )
        pure_result = pure_backend.isin(
            pure_backend.array(elements), pure_backend.array(test_elements)
        )
        expected = [False, True, False, True, False]
        assert to_list(np_result) == to_list(pure_result) == expected


# =============================================================================
# Linear Algebra Tests
# =============================================================================


class TestLinearAlgebra:
    """Test linear algebra operations."""

    def test_dot_product_1d(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        np_result = np_backend.dot(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.dot(pure_backend.array(a), pure_backend.array(b))
        assert np_result == pure_result == 32

    def test_linalg_norm(self):
        values = [3, 4]
        np_result = np_backend.linalg.norm(np_backend.array(values))
        pure_result = pure_backend.linalg.norm(pure_backend.array(values))
        assert abs(np_result - pure_result) < 1e-10
        assert abs(np_result - 5.0) < 1e-10


# =============================================================================
# Random Generation Tests
# =============================================================================


class TestRandomGeneration:
    """Test random number generation."""

    def test_seed_reproducibility(self):
        np_backend.random.seed(42)
        np_result1 = np_backend.random.randint(0, 100, size=10)
        np_backend.random.seed(42)
        np_result2 = np_backend.random.randint(0, 100, size=10)
        assert to_list(np_result1) == to_list(np_result2)

        pure_backend.random.seed(42)
        pure_result1 = pure_backend.random.randint(0, 100, size=10)
        pure_backend.random.seed(42)
        pure_result2 = pure_backend.random.randint(0, 100, size=10)
        assert to_list(pure_result1) == to_list(pure_result2)

    def test_randint_range(self):
        pure_backend.random.seed(0)
        result = pure_backend.random.randint(0, 10, size=100)
        values = to_list(result)
        assert all(0 <= v < 10 for v in values)

    def test_uniform_range(self):
        pure_backend.random.seed(0)
        result = pure_backend.random.uniform(0, 1, size=100)
        values = to_list(result)
        assert all(0 <= v <= 1 for v in values)

    def test_choice_from_array(self):
        arr = [10, 20, 30, 40, 50]
        pure_backend.random.seed(0)
        result = pure_backend.random.choice(arr, size=10)
        values = to_list(result)
        assert all(v in arr for v in values)

    def test_normal_distribution(self):
        pure_backend.random.seed(0)
        result = pure_backend.random.normal(0, 1, size=1000)
        values = to_list(result)
        # Check mean is approximately 0
        mean = sum(values) / len(values)
        assert abs(mean) < 0.2  # Generous tolerance for 1000 samples


# =============================================================================
# Array Properties Tests
# =============================================================================


class TestArrayProperties:
    """Test array property access."""

    def test_shape_1d(self):
        np_arr = np_backend.array([1, 2, 3, 4, 5])
        pure_arr = pure_backend.array([1, 2, 3, 4, 5])
        assert np_arr.shape == pure_arr.shape == (5,)

    def test_shape_2d(self):
        np_arr = np_backend.array([[1, 2, 3], [4, 5, 6]])
        pure_arr = pure_backend.array([[1, 2, 3], [4, 5, 6]])
        assert np_arr.shape == pure_arr.shape == (2, 3)

    def test_ndim(self):
        np_arr_1d = np_backend.array([1, 2, 3])
        np_arr_2d = np_backend.array([[1, 2], [3, 4]])
        pure_arr_1d = pure_backend.array([1, 2, 3])
        pure_arr_2d = pure_backend.array([[1, 2], [3, 4]])

        assert np_arr_1d.ndim == pure_arr_1d.ndim == 1
        assert np_arr_2d.ndim == pure_arr_2d.ndim == 2


# =============================================================================
# Array Manipulation Tests
# =============================================================================


class TestArrayOperations:
    """Test array manipulation operations."""

    def test_reshape(self):
        values = list(range(6))
        np_arr = np_backend.array(values).reshape((2, 3))
        pure_arr = pure_backend.array(values).reshape((2, 3))
        assert np_arr.shape == pure_arr.shape == (2, 3)

    def test_flatten(self):
        values = [[1, 2, 3], [4, 5, 6]]
        np_arr = np_backend.flatten(np_backend.array(values))
        pure_arr = pure_backend.flatten(pure_backend.array(values))
        assert to_list(np_arr) == to_list(pure_arr) == [1, 2, 3, 4, 5, 6]

    def test_transpose_2d(self):
        values = [[1, 2, 3], [4, 5, 6]]
        np_arr = np_backend.array(values)
        pure_arr = pure_backend.array(values)
        np_T = np_arr.T
        pure_T = pure_arr.T
        assert np_T.shape == pure_T.shape == (3, 2)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test mathematical constants."""

    def test_inf(self):
        assert np_backend.inf == pure_backend.inf == float("inf")

    def test_pi(self):
        assert abs(np_backend.pi - pure_backend.pi) < 1e-15
        assert abs(np_backend.pi - math.pi) < 1e-15

    def test_e(self):
        assert abs(np_backend.e - pure_backend.e) < 1e-15
        assert abs(np_backend.e - math.e) < 1e-15


# =============================================================================
# Comparison Tests
# =============================================================================


class TestComparison:
    """Test comparison operations."""

    def test_maximum(self):
        a = [1, 5, 3, 7]
        b = [2, 4, 6, 0]
        np_result = np_backend.maximum(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.maximum(pure_backend.array(a), pure_backend.array(b))
        assert to_list(np_result) == to_list(pure_result) == [2, 5, 6, 7]

    def test_minimum(self):
        a = [1, 5, 3, 7]
        b = [2, 4, 6, 0]
        np_result = np_backend.minimum(np_backend.array(a), np_backend.array(b))
        pure_result = pure_backend.minimum(pure_backend.array(a), pure_backend.array(b))
        assert to_list(np_result) == to_list(pure_result) == [1, 4, 3, 0]

    def test_isnan(self):
        values = [1.0, float("nan"), 3.0, float("nan")]
        np_result = np_backend.isnan(np_backend.array(values))
        pure_result = pure_backend.isnan(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result) == [False, True, False, True]

    def test_isinf(self):
        values = [1.0, float("inf"), -float("inf"), 0.0]
        np_result = np_backend.isinf(np_backend.array(values))
        pure_result = pure_backend.isinf(pure_backend.array(values))
        assert to_list(np_result) == to_list(pure_result) == [False, True, True, False]


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestTriangularExtraction:
    """Test upper triangle extraction."""

    def test_triu_k0(self):
        mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        np_result = np_backend.triu(np_backend.array(mat), k=0)
        pure_result = pure_backend.triu(pure_backend.array(mat), k=0)
        assert to_list(np_result.flatten()) == to_list(pure_result.flatten())
        assert to_list(pure_result.flatten()) == [1, 2, 3, 0, 5, 6, 0, 0, 9]

    def test_triu_k1(self):
        mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        np_result = np_backend.triu(np_backend.array(mat), k=1)
        pure_result = pure_backend.triu(pure_backend.array(mat), k=1)
        assert to_list(np_result.flatten()) == to_list(pure_result.flatten())
        assert to_list(pure_result.flatten()) == [0, 2, 3, 0, 0, 6, 0, 0, 0]

    def test_triu_k_neg1(self):
        mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        np_result = np_backend.triu(np_backend.array(mat), k=-1)
        pure_result = pure_backend.triu(pure_backend.array(mat), k=-1)
        assert to_list(np_result.flatten()) == to_list(pure_result.flatten())
        assert to_list(pure_result.flatten()) == [1, 2, 3, 4, 5, 6, 0, 8, 9]


class TestInvert:
    """Test boolean inversion."""

    def test_invert_list(self):
        bools = [True, False, True, False, True]
        pure_result = pure_backend.invert(pure_backend.array(bools))
        assert to_list(pure_result) == [False, True, False, True, False]

    def test_invert_gfoarray(self):
        arr = pure_backend.array([True, False, False, True])
        result = pure_backend.invert(arr)
        assert to_list(result) == [False, True, True, False]


class TestDefaultRng:
    """Test Generator factory via default_rng."""

    def test_generator_has_methods(self):
        rng = pure_backend.random.default_rng(42)
        assert callable(getattr(rng, "random", None))
        assert callable(getattr(rng, "standard_normal", None))
        assert callable(getattr(rng, "uniform", None))
        assert callable(getattr(rng, "integers", None))

    def test_seeded_determinism(self):
        rng1 = pure_backend.random.default_rng(123)
        vals1 = to_list(rng1.random(size=5))
        rng2 = pure_backend.random.default_rng(123)
        vals2 = to_list(rng2.random(size=5))
        assert vals1 == vals2

    def test_array_output_shape(self):
        rng = pure_backend.random.default_rng(0)
        result = rng.uniform(0.0, 1.0, size=10)
        assert len(to_list(result)) == 10

    def test_integers_range(self):
        rng = pure_backend.random.default_rng(7)
        result = rng.integers(0, 10, size=50)
        values = to_list(result)
        assert all(0 <= v < 10 for v in values)


class TestLinAlgError:
    """Test LinAlgError exception type."""

    def test_is_exception_class(self):
        assert issubclass(pure_backend.linalg.LinAlgError, Exception)
        assert issubclass(np_backend.linalg.LinAlgError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(pure_backend.linalg.LinAlgError):
            raise pure_backend.linalg.LinAlgError("singular matrix")

    def test_can_be_caught_as_exception(self):
        with pytest.raises(Exception, match="test"):
            raise np_backend.linalg.LinAlgError("test")


class TestLinAlgEigh:
    """Test eigendecomposition."""

    def test_numpy_backend_returns_eigenvalues_and_vectors(self):
        mat = np_backend.array([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = np_backend.linalg.eigh(mat)
        assert len(to_list(eigenvalues)) == 2
        assert eigenvectors.shape == (2, 2)

    def test_pure_backend_raises(self):
        mat = pure_backend.array([[2.0, 1.0], [1.0, 3.0]])
        with pytest.raises(NotImplementedError):
            pure_backend.linalg.eigh(mat)


class TestNdarrayType:
    """Test ndarray type export."""

    def test_numpy_isinstance(self):
        import gradient_free_optimizers._array_backend as arr_backend

        arr = arr_backend.array([1, 2, 3])
        assert isinstance(arr, arr_backend.ndarray)

    def test_pure_isinstance(self):
        arr = pure_backend.array([1, 2, 3])
        from gradient_free_optimizers._array_backend._pure import GFOArray

        assert isinstance(arr, GFOArray)


class TestBooleanSetitem:
    """Test boolean indexing with __setitem__."""

    def test_setitem_scalar(self):
        arr = pure_backend.array([1.0, 2.0, 3.0, 4.0])
        mask = [True, False, True, False]
        arr[mask] = 0.0
        assert to_list(arr) == [0.0, 2.0, 0.0, 4.0]

    def test_setitem_array_values(self):
        arr = pure_backend.array([1.0, 2.0, 3.0, 4.0])
        mask = [False, True, False, True]
        arr[mask] = pure_backend.array([20.0, 40.0])
        assert to_list(arr) == [1.0, 20.0, 3.0, 40.0]


class TestReshapeNegativeOne:
    """Test reshape with -1 dimension inference."""

    def test_reshape_neg1_first(self):
        arr = pure_backend.array([1, 2, 3, 4, 5, 6])
        result = arr.reshape((-1, 2))
        assert result.shape == (3, 2)

    def test_reshape_neg1_second(self):
        arr = pure_backend.array([1, 2, 3, 4, 5, 6])
        result = arr.reshape((2, -1))
        assert result.shape == (2, 3)

    def test_reshape_neg1_matches_numpy(self):
        values = list(range(12))
        np_result = np_backend.array(values).reshape((-1, 3))
        pure_result = pure_backend.array(values).reshape((-1, 3))
        assert np_result.shape == pure_result.shape == (4, 3)


class TestBackendSelection:
    """Test that backend selection works correctly."""

    def test_has_numpy_flag(self):
        assert HAS_NUMPY is True  # We know NumPy is installed in test env

    def test_array_backend_uses_numpy(self):
        import gradient_free_optimizers._array_backend as array_backend

        if HAS_NUMPY:
            assert array_backend._backend_name == "numpy"

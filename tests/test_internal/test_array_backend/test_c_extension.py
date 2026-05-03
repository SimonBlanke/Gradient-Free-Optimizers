"""Tests for the C extension array operations.

Verifies that C-accelerated functions produce identical results
to the pure Python implementations.
"""

import array
import math

import pytest

from gradient_free_optimizers._array_backend import HAS_C_EXTENSION

pytestmark = pytest.mark.skipif(not HAS_C_EXTENSION, reason="C extension not compiled")

if HAS_C_EXTENSION:
    from gradient_free_optimizers._array_backend import _fast_ops as fo


def _arr(*vals):
    return array.array("d", vals)


def _from_bytes(raw):
    result = array.array("d")
    result.frombytes(raw)
    return list(result)


class TestVecArithmetic:
    def test_add(self):
        r = _from_bytes(fo.vec_add(_arr(1, 2, 3), _arr(4, 5, 6)))
        assert r == [5.0, 7.0, 9.0]

    def test_sub(self):
        r = _from_bytes(fo.vec_sub(_arr(10, 20, 30), _arr(1, 2, 3)))
        assert r == [9.0, 18.0, 27.0]

    def test_mul(self):
        r = _from_bytes(fo.vec_mul(_arr(2, 3, 4), _arr(5, 6, 7)))
        assert r == [10.0, 18.0, 28.0]

    def test_add_scalar(self):
        r = _from_bytes(fo.vec_add_scalar(_arr(1, 2, 3), 10.0))
        assert r == [11.0, 12.0, 13.0]

    def test_mul_scalar(self):
        r = _from_bytes(fo.vec_mul_scalar(_arr(1, 2, 3), 0.5))
        assert r == [0.5, 1.0, 1.5]

    def test_neg(self):
        r = _from_bytes(fo.vec_neg(_arr(1, -2, 3)))
        assert r == [-1.0, 2.0, -3.0]


class TestVecMath:
    def test_exp(self):
        r = _from_bytes(fo.vec_exp(_arr(0, 1, 2)))
        assert abs(r[0] - 1.0) < 1e-10
        assert abs(r[1] - math.e) < 1e-10

    def test_log(self):
        r = _from_bytes(fo.vec_log(_arr(1, math.e, math.e**2)))
        assert abs(r[0] - 0.0) < 1e-10
        assert abs(r[1] - 1.0) < 1e-10
        assert abs(r[2] - 2.0) < 1e-10

    def test_sqrt(self):
        r = _from_bytes(fo.vec_sqrt(_arr(0, 1, 4, 9)))
        assert r == [0.0, 1.0, 2.0, 3.0]

    def test_clip(self):
        r = _from_bytes(fo.vec_clip(_arr(-1, 0, 5, 10, 15), 0.0, 10.0))
        assert r == [0.0, 0.0, 5.0, 10.0, 10.0]


class TestReductions:
    def test_sum(self):
        assert fo.vec_sum(_arr(1, 2, 3, 4)) == 10.0

    def test_sum_single(self):
        assert fo.vec_sum(_arr(42.0)) == 42.0

    def test_argmax(self):
        assert fo.vec_argmax(_arr(1, 5, 3, 2)) == 1

    def test_argmax_first_element(self):
        assert fo.vec_argmax(_arr(99, 1, 2)) == 0

    def test_argmax_last_element(self):
        assert fo.vec_argmax(_arr(1, 2, 99)) == 2

    def test_dot(self):
        assert fo.vec_dot(_arr(1, 2, 3), _arr(4, 5, 6)) == 32.0


class TestMatMul:
    def test_identity(self):
        I = _arr(1, 0, 0, 1)
        v = _arr(3, 7, 2, 5)
        r = _from_bytes(fo.mat_mul(I, v, 2, 2, 2))
        assert r == [3.0, 7.0, 2.0, 5.0]

    def test_2x3_times_3x2(self):
        A = _arr(1, 2, 3, 4, 5, 6)
        B = _arr(7, 8, 9, 10, 11, 12)
        r = _from_bytes(fo.mat_mul(A, B, 2, 3, 2))
        assert r == [58.0, 64.0, 139.0, 154.0]


class TestCGFOArrayIntegration:
    """Test that _CGFOArray accelerates operations correctly."""

    def test_c_extension_module_level_functions(self):
        from gradient_free_optimizers._array_backend import _c_extension as ce

        a = ce.array([1.0, 2.0, 3.0])
        b = ce.array([4.0, 5.0, 6.0])

        result = a + b
        assert list(result._data) == [5.0, 7.0, 9.0]

        result = a * 2.0
        assert list(result._data) == [2.0, 4.0, 6.0]

        result = ce.exp(ce.array([0.0]))
        assert abs(list(result._data)[0] - 1.0) < 1e-10

        assert ce.sum(a) == 6.0
        assert ce.argmax(b) == 2

    def test_c_extension_clip(self):
        from gradient_free_optimizers._array_backend import _c_extension as ce

        result = ce.clip(ce.array([1.0, 5.0, 10.0]), 2.0, 8.0)
        assert list(result._data) == [2.0, 5.0, 8.0]

    def test_c_extension_dot(self):
        from gradient_free_optimizers._array_backend import _c_extension as ce

        a = ce.array([1.0, 2.0, 3.0])
        b = ce.array([4.0, 5.0, 6.0])
        assert ce.dot(a, b) == 32.0

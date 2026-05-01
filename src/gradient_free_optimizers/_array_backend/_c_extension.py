"""C-accelerated array backend using the compiled _fast_ops module.

Wraps the pure Python GFOArray class but accelerates hot-path operations
(arithmetic, math functions, reductions) through C loops. Falls back to
pure Python for operations not covered by the C extension.

This module re-exports everything from _pure.py, then overrides the
performance-critical functions with C-accelerated versions.
"""

import array as _array_mod

from . import _fast_ops
from ._pure import *  # noqa: F401, F403
from ._pure import _DOUBLE, GFOArray

_frombytes = _array_mod.array.frombytes


def _c_result(raw_bytes, shape):
    data = _array_mod.array(_DOUBLE)
    _frombytes(data, raw_bytes)
    return GFOArray._from_raw(data, shape)


class _CGFOArray(GFOArray):
    """GFOArray subclass with C-accelerated arithmetic."""

    def __add__(self, other):
        if (
            isinstance(other, GFOArray)
            and isinstance(self._data, _array_mod.array)
            and isinstance(other._data, _array_mod.array)
        ):
            return _c_result(_fast_ops.vec_add(self._data, other._data), self._shape)
        if isinstance(other, int | float) and isinstance(self._data, _array_mod.array):
            return _c_result(
                _fast_ops.vec_add_scalar(self._data, float(other)),
                self._shape,
            )
        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if (
            isinstance(other, GFOArray)
            and isinstance(self._data, _array_mod.array)
            and isinstance(other._data, _array_mod.array)
        ):
            return _c_result(_fast_ops.vec_sub(self._data, other._data), self._shape)
        if isinstance(other, int | float) and isinstance(self._data, _array_mod.array):
            return _c_result(
                _fast_ops.vec_add_scalar(self._data, -float(other)),
                self._shape,
            )
        return super().__sub__(other)

    def __mul__(self, other):
        if (
            isinstance(other, GFOArray)
            and isinstance(self._data, _array_mod.array)
            and isinstance(other._data, _array_mod.array)
        ):
            return _c_result(_fast_ops.vec_mul(self._data, other._data), self._shape)
        if isinstance(other, int | float) and isinstance(self._data, _array_mod.array):
            return _c_result(
                _fast_ops.vec_mul_scalar(self._data, float(other)),
                self._shape,
            )
        return super().__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        if isinstance(self._data, _array_mod.array):
            return _c_result(_fast_ops.vec_neg(self._data), self._shape)
        return super().__neg__()

    def sum(self, axis=None):
        if axis is None and isinstance(self._data, _array_mod.array):
            return _fast_ops.vec_sum(self._data)
        return super().sum(axis=axis)

    def argmax(self, axis=None):
        if axis is None and isinstance(self._data, _array_mod.array):
            return _fast_ops.vec_argmax(self._data)
        return super().argmax(axis=axis)

    def __matmul__(self, other):
        if not isinstance(other, GFOArray):
            other = GFOArray(other)
        if (
            self._ndim == 1
            and other._ndim == 1
            and isinstance(self._data, _array_mod.array)
            and isinstance(other._data, _array_mod.array)
        ):
            return _fast_ops.vec_dot(self._data, other._data)
        if (
            self._ndim == 2
            and other._ndim == 2
            and isinstance(self._data, _array_mod.array)
            and isinstance(other._data, _array_mod.array)
        ):
            m, k = self._shape
            k2, n = other._shape
            if k == k2:
                return _c_result(
                    _fast_ops.mat_mul(self._data, other._data, m, k, n),
                    (m, n),
                )
        return super().__matmul__(other)


def _c_buf(raw_bytes):
    data = _array_mod.array(_DOUBLE)
    _frombytes(data, raw_bytes)
    return data


def array(data, dtype=None):
    """Create a C-accelerated GFOArray."""
    base = GFOArray(data, dtype=dtype)
    return _CGFOArray._from_raw(base._data, base._shape)


def zeros(shape, dtype=float):
    from ._pure import zeros as _pure_zeros

    base = _pure_zeros(shape, dtype)
    if isinstance(base._data, _array_mod.array):
        return _CGFOArray._from_raw(base._data, base._shape)
    return base


def ones(shape, dtype=float):
    from ._pure import ones as _pure_ones

    base = _pure_ones(shape, dtype)
    if isinstance(base._data, _array_mod.array):
        return _CGFOArray._from_raw(base._data, base._shape)
    return base


def empty(shape, dtype=float):
    from ._pure import empty as _pure_empty

    base = _pure_empty(shape, dtype)
    if isinstance(base._data, _array_mod.array):
        return _CGFOArray._from_raw(base._data, base._shape)
    return base


def full(shape, fill_value, dtype=None):
    from ._pure import full as _pure_full

    base = _pure_full(shape, fill_value, dtype)
    if isinstance(base._data, _array_mod.array):
        return _CGFOArray._from_raw(base._data, base._shape)
    return base


def exp(x):
    if isinstance(x, GFOArray) and isinstance(x._data, _array_mod.array):
        return _c_result(_fast_ops.vec_exp(x._data), x._shape)
    from ._pure import exp as _pure_exp

    return _pure_exp(x)


def log(x):
    if isinstance(x, GFOArray) and isinstance(x._data, _array_mod.array):
        return _c_result(_fast_ops.vec_log(x._data), x._shape)
    from ._pure import log as _pure_log

    return _pure_log(x)


def sqrt(x):
    if isinstance(x, GFOArray) and isinstance(x._data, _array_mod.array):
        return _c_result(_fast_ops.vec_sqrt(x._data), x._shape)
    from ._pure import sqrt as _pure_sqrt

    return _pure_sqrt(x)


def clip(x, a_min, a_max):
    if (
        isinstance(x, GFOArray)
        and isinstance(x._data, _array_mod.array)
        and isinstance(a_min, int | float)
        and isinstance(a_max, int | float)
    ):
        return _c_result(
            _fast_ops.vec_clip(x._data, float(a_min), float(a_max)),
            x._shape,
        )
    from ._pure import clip as _pure_clip

    return _pure_clip(x, a_min, a_max)


def sum(x, axis=None):
    if (
        isinstance(x, GFOArray)
        and axis is None
        and isinstance(x._data, _array_mod.array)
    ):
        return _fast_ops.vec_sum(x._data)
    from ._pure import sum as _pure_sum

    return _pure_sum(x, axis)


def argmax(x, axis=None):
    if (
        isinstance(x, GFOArray)
        and axis is None
        and isinstance(x._data, _array_mod.array)
    ):
        return _fast_ops.vec_argmax(x._data)
    from ._pure import argmax as _pure_argmax

    return _pure_argmax(x, axis)


def dot(a, b):
    if (
        isinstance(a, GFOArray)
        and isinstance(b, GFOArray)
        and isinstance(a._data, _array_mod.array)
        and isinstance(b._data, _array_mod.array)
        and len(a._data) == len(b._data)
    ):
        return _fast_ops.vec_dot(a._data, b._data)
    from ._pure import dot as _pure_dot

    return _pure_dot(a, b)

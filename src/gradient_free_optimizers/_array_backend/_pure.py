"""Pure Python array backend optimized for performance.

Uses array.array('d') for contiguous float64 storage and C-level
operator/map for element-wise operations. Falls back to list storage
for non-numeric data (booleans, objects).

2D arrays are stored flat with stride-based indexing.
"""

import array as _array_mod
import builtins
import itertools
import math
import operator
import random as py_random
from typing import Any, Union

_sum = builtins.sum
_min = builtins.min
_max = builtins.max
_abs = builtins.abs
_all = builtins.all
_any = builtins.any
_round = builtins.round

_add = operator.add
_sub = operator.sub
_mul = operator.mul
_truediv = operator.truediv
_floordiv = operator.floordiv
_mod = operator.mod
_neg = operator.neg
_pow_op = operator.pow

_repeat = itertools.repeat

_m_exp = math.exp
_m_log = math.log
_m_log10 = math.log10
_m_sqrt = math.sqrt
_m_sin = math.sin
_m_cos = math.cos
_m_floor = math.floor
_m_ceil = math.ceil
_m_isnan = math.isnan
_m_isinf = math.isinf
_m_isfinite = math.isfinite

_DOUBLE = "d"

inf = float("inf")
pi = math.pi
e = math.e
nan = float("nan")

ArrayLike = Union["GFOArray", list, tuple, int, float]
Shape = int | tuple[int, ...]

int32 = int
int64 = int
float32 = float
float64 = float


class GFOArray:
    """Pure Python array with contiguous float64 storage.

    Uses array.array('d') for numeric data and Python list for
    non-numeric data (booleans, objects). 2D arrays are stored
    flat with stride-based row access.
    """

    __slots__ = ("_data", "_shape", "_ndim")

    def __init__(self, data: Any, dtype: type | None = None):
        if isinstance(data, GFOArray):
            if isinstance(data._data, _array_mod.array):
                self._data = _array_mod.array(_DOUBLE, data._data)
            else:
                self._data = data._data.copy()
            self._shape = data._shape
            self._ndim = data._ndim
        elif isinstance(data, _array_mod.array):
            self._data = _array_mod.array(_DOUBLE, data)
            self._shape = (len(data),)
            self._ndim = 1
        elif isinstance(data, list | tuple):
            if len(data) == 0:
                self._data = _array_mod.array(_DOUBLE)
                self._shape = (0,)
                self._ndim = 1
            elif isinstance(data[0], list | tuple | GFOArray):
                nrows = len(data)
                row0 = data[0]
                ncols = (
                    len(row0._data)
                    if isinstance(row0, GFOArray) and row0._ndim == 1
                    else len(row0)
                )
                flat = []
                for row in data:
                    if isinstance(row, GFOArray):
                        flat.extend(row._data)
                    else:
                        flat.extend(row)
                try:
                    self._data = _array_mod.array(_DOUBLE, flat)
                except TypeError:
                    self._data = flat
                self._shape = (nrows, ncols)
                self._ndim = 2
            elif type(data[0]) is bool:
                self._data = list(data)
                self._shape = (len(data),)
                self._ndim = 1
            else:
                try:
                    self._data = _array_mod.array(_DOUBLE, data)
                except TypeError:
                    self._data = list(data)
                self._shape = (len(data),)
                self._ndim = 1
        elif isinstance(data, int | float):
            self._data = _array_mod.array(_DOUBLE, [data])
            self._shape = (1,)
            self._ndim = 1
        else:
            self._data = [data]
            self._shape = (1,)
            self._ndim = 1

        if dtype is not None:
            self._apply_dtype(dtype)

    @classmethod
    def _from_raw(cls, data, shape):
        """Fast construction from pre-built data. No copying or type checking."""
        obj = object.__new__(cls)
        obj._data = data
        obj._shape = shape
        obj._ndim = len(shape)
        return obj

    def _apply_dtype(self, dtype):
        if dtype is bool:
            self._data = [bool(x) for x in self._data]
        elif dtype is object:
            self._data = list(self._data)
        elif dtype is int or dtype is int64 or dtype is int32:
            if isinstance(self._data, _array_mod.array):
                self._data = _array_mod.array(
                    _DOUBLE, (float(int(x)) for x in self._data)
                )
            else:
                self._data = [int(x) for x in self._data]
        elif dtype is float or dtype is float64 or dtype is float32:
            if not isinstance(self._data, _array_mod.array):
                try:
                    self._data = _array_mod.array(_DOUBLE, self._data)
                except TypeError:
                    self._data = [float(x) for x in self._data]
        else:
            self._data = [dtype(x) for x in self._data]

    def _get_flat(self):
        return list(self._data)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def size(self):
        s = 1
        for d in self._shape:
            s *= d
        return s

    @property
    def dtype(self):
        if isinstance(self._data, _array_mod.array):
            return float64
        if len(self._data) == 0:
            return float64
        return type(self._data[0])

    @property
    def T(self):
        return self.transpose()

    def __len__(self):
        return self._shape[0]

    def __iter__(self):
        if self._ndim == 1:
            return iter(self._data)
        ncols = self._shape[1]
        data = self._data
        return (
            GFOArray._from_raw(data[i * ncols : (i + 1) * ncols], (ncols,))
            for i in range(self._shape[0])
        )

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if self._ndim == 2:
                return self._getitem_2d(idx)

        if isinstance(idx, int):
            if self._ndim == 1:
                return self._data[idx]
            ncols = self._shape[1]
            r = idx if idx >= 0 else self._shape[0] + idx
            start = r * ncols
            return GFOArray._from_raw(self._data[start : start + ncols], (ncols,))

        if isinstance(idx, slice):
            if self._ndim == 1:
                sliced = self._data[idx]
                return GFOArray._from_raw(sliced, (len(sliced),))
            return self._getitem_2d_slice(idx)

        if isinstance(idx, list | GFOArray):
            return self._getitem_fancy(idx)

        return self._data[idx]

    def _getitem_2d(self, idx):
        row_idx, col_idx = idx
        ncols = self._shape[1]

        if isinstance(row_idx, int) and isinstance(col_idx, int):
            r = row_idx if row_idx >= 0 else self._shape[0] + row_idx
            c = col_idx if col_idx >= 0 else ncols + col_idx
            return self._data[r * ncols + c]

        if isinstance(row_idx, slice):
            row_range = range(*row_idx.indices(self._shape[0]))
            if isinstance(col_idx, int):
                c = col_idx if col_idx >= 0 else ncols + col_idx
                result = [self._data[r * ncols + c] for r in row_range]
                n = len(result)
                if isinstance(self._data, _array_mod.array):
                    return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (n,))
                return GFOArray._from_raw(result, (n,))
            col_range = range(*col_idx.indices(ncols))
            result = []
            for r in row_range:
                base = r * ncols
                for c in col_range:
                    result.append(self._data[base + c])
            nr, nc = len(row_range), len(col_range)
            if isinstance(self._data, _array_mod.array):
                return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (nr, nc))
            return GFOArray._from_raw(result, (nr, nc))

        if isinstance(row_idx, int):
            r = row_idx if row_idx >= 0 else self._shape[0] + row_idx
            base = r * ncols
            if isinstance(col_idx, slice):
                col_range = range(*col_idx.indices(ncols))
                result = [self._data[base + c] for c in col_range]
                n = len(result)
                if isinstance(self._data, _array_mod.array):
                    return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (n,))
                return GFOArray._from_raw(result, (n,))
            if isinstance(col_idx, list | GFOArray):
                cols = list(col_idx) if isinstance(col_idx, GFOArray) else col_idx
                result = [self._data[base + c] for c in cols]
                n = len(result)
                if isinstance(self._data, _array_mod.array):
                    return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (n,))
                return GFOArray._from_raw(result, (n,))

    def _getitem_2d_slice(self, idx):
        ncols = self._shape[1]
        row_range = range(*idx.indices(self._shape[0]))
        nrows = len(row_range)
        if (
            isinstance(self._data, _array_mod.array)
            and row_range.step == 1
            and nrows > 0
        ):
            start = row_range.start * ncols
            end = (row_range.start + nrows) * ncols
            return GFOArray._from_raw(self._data[start:end], (nrows, ncols))
        result = []
        for r in row_range:
            base = r * ncols
            result.extend(self._data[base : base + ncols])
        if isinstance(self._data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (nrows, ncols))
        return GFOArray._from_raw(result, (nrows, ncols))

    def _getitem_fancy(self, idx):
        idx_list = list(idx) if isinstance(idx, GFOArray) else idx
        if not idx_list:
            if isinstance(self._data, _array_mod.array):
                return GFOArray._from_raw(_array_mod.array(_DOUBLE), (0,))
            return GFOArray._from_raw([], (0,))

        is_bool = isinstance(idx_list[0], bool)

        if not is_bool and isinstance(idx_list[0], float):
            idx_list = [int(i) for i in idx_list]

        if self._ndim == 1:
            if is_bool:
                result = [self._data[i] for i, b in enumerate(idx_list) if b]
            else:
                result = [self._data[i] for i in idx_list]
            n = len(result)
            if isinstance(self._data, _array_mod.array):
                return GFOArray._from_raw(
                    _array_mod.array(_DOUBLE, result)
                    if result
                    else _array_mod.array(_DOUBLE),
                    (n,),
                )
            return GFOArray._from_raw(result, (n,))

        ncols = self._shape[1]
        if is_bool:
            selected = [i for i, b in enumerate(idx_list) if b]
        else:
            selected = idx_list
        nrows = len(selected)
        if isinstance(self._data, _array_mod.array):
            result = _array_mod.array(_DOUBLE)
            for r in selected:
                base = r * ncols
                result.extend(self._data[base : base + ncols])
            return GFOArray._from_raw(result, (nrows, ncols))
        result = []
        for r in selected:
            base = r * ncols
            result.extend(self._data[base : base + ncols])
        return GFOArray._from_raw(result, (nrows, ncols))

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            if self._ndim == 1:
                self._data[idx] = value
            else:
                ncols = self._shape[1]
                r = idx if idx >= 0 else self._shape[0] + idx
                start = r * ncols
                if isinstance(value, GFOArray):
                    src = value._data
                elif isinstance(value, list | tuple):
                    src = value
                else:
                    for i in range(ncols):
                        self._data[start + i] = value
                    return
                if isinstance(self._data, _array_mod.array):
                    self._data[start : start + ncols] = (
                        src
                        if isinstance(src, _array_mod.array)
                        else _array_mod.array(_DOUBLE, src)
                    )
                else:
                    self._data[start : start + ncols] = list(src)
        elif isinstance(idx, tuple) and self._ndim == 2:
            row_idx, col_idx = idx
            ncols = self._shape[1]
            r = row_idx if row_idx >= 0 else self._shape[0] + row_idx
            c = col_idx if col_idx >= 0 else ncols + col_idx
            self._data[r * ncols + c] = value
        elif isinstance(idx, list | GFOArray):
            idx_list = list(idx) if isinstance(idx, GFOArray) else idx
            if idx_list and isinstance(idx_list[0], bool):
                if isinstance(value, GFOArray):
                    vals = list(value._data)
                    vi = 0
                    for i, b in enumerate(idx_list):
                        if b:
                            self._data[i] = vals[vi]
                            vi += 1
                elif isinstance(value, list | tuple):
                    vi = 0
                    for i, b in enumerate(idx_list):
                        if b:
                            self._data[i] = value[vi]
                            vi += 1
                else:
                    for i, b in enumerate(idx_list):
                        if b:
                            self._data[i] = value

    def __repr__(self):
        if self._ndim == 1:
            return f"GFOArray({list(self._data)})"
        ncols = self._shape[1]
        rows = [
            list(self._data[i * ncols : (i + 1) * ncols]) for i in range(self._shape[0])
        ]
        return f"GFOArray({rows})"

    def __str__(self):
        if self._ndim == 1:
            return str(list(self._data))
        ncols = self._shape[1]
        rows = [
            list(self._data[i * ncols : (i + 1) * ncols]) for i in range(self._shape[0])
        ]
        return str(rows)

    def _broadcast_other(self, other):
        """Tile 1D other._data to match self's 2D flat layout."""
        if self._ndim == 2 and other._ndim == 1:
            ncols = self._shape[1]
            nrows = self._shape[0]
            if len(other._data) == ncols:
                od = other._data
                if isinstance(od, _array_mod.array):
                    return od * nrows
                return od * nrows
        return other._data

    def _binop(self, other, op):
        if isinstance(other, GFOArray):
            other_data = self._broadcast_other(other)
            if isinstance(self._data, _array_mod.array) and isinstance(
                other_data, _array_mod.array
            ):
                return GFOArray._from_raw(
                    _array_mod.array(_DOUBLE, map(op, self._data, other_data)),
                    self._shape,
                )
            return GFOArray._from_raw(
                list(map(op, self._data, other_data)), self._shape
            )
        if isinstance(other, int | float):
            if isinstance(self._data, _array_mod.array):
                return GFOArray._from_raw(
                    _array_mod.array(_DOUBLE, map(op, self._data, _repeat(other))),
                    self._shape,
                )
            return GFOArray._from_raw(
                list(map(op, self._data, _repeat(other))), self._shape
            )
        return GFOArray._from_raw(list(map(op, self._data, other)), self._shape)

    def _rbinop(self, other, op):
        """Reverse binary op: other op self."""
        if isinstance(self._data, _array_mod.array):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, map(op, _repeat(other), self._data)),
                self._shape,
            )
        return GFOArray._from_raw(
            list(map(op, _repeat(other), self._data)), self._shape
        )

    def _cmpop(self, other, op):
        if isinstance(other, GFOArray):
            return GFOArray._from_raw(
                list(map(op, self._data, other._data)), self._shape
            )
        if isinstance(other, int | float):
            return GFOArray._from_raw(
                list(map(op, self._data, _repeat(other))), self._shape
            )
        return GFOArray._from_raw(list(map(op, self._data, other)), self._shape)

    def __add__(self, other):
        return self._binop(other, _add)

    def __radd__(self, other):
        return self._binop(other, _add)

    def __sub__(self, other):
        return self._binop(other, _sub)

    def __rsub__(self, other):
        return self._rbinop(other, _sub)

    def __mul__(self, other):
        return self._binop(other, _mul)

    def __rmul__(self, other):
        return self._binop(other, _mul)

    def __truediv__(self, other):
        return self._binop(other, _truediv)

    def __rtruediv__(self, other):
        return self._rbinop(other, _truediv)

    def __floordiv__(self, other):
        return self._binop(other, _floordiv)

    def __pow__(self, other):
        return self._binop(other, _pow_op)

    def __mod__(self, other):
        return self._binop(other, _mod)

    def __neg__(self):
        if isinstance(self._data, _array_mod.array):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, map(_neg, self._data)), self._shape
            )
        return GFOArray._from_raw(list(map(_neg, self._data)), self._shape)

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        if isinstance(self._data, _array_mod.array):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, map(_abs, self._data)), self._shape
            )
        return GFOArray._from_raw(list(map(_abs, self._data)), self._shape)

    def __invert__(self):
        return GFOArray._from_raw([not x for x in self._data], self._shape)

    def __and__(self, other):
        if isinstance(other, GFOArray):
            return GFOArray._from_raw(
                [a and b for a, b in zip(self._data, other._data)], self._shape
            )
        return GFOArray._from_raw([a and other for a in self._data], self._shape)

    def __or__(self, other):
        if isinstance(other, GFOArray):
            return GFOArray._from_raw(
                [a or b for a, b in zip(self._data, other._data)], self._shape
            )
        return GFOArray._from_raw([a or other for a in self._data], self._shape)

    def __matmul__(self, other):
        if not isinstance(other, GFOArray):
            other = GFOArray(other)

        if self._ndim == 1 and other._ndim == 1:
            return _sum(map(_mul, self._data, other._data))

        if self._ndim == 2 and other._ndim == 1:
            m, k = self._shape
            sd, od = self._data, other._data
            result = _array_mod.array(
                _DOUBLE,
                (_sum(sd[i * k + j] * od[j] for j in range(k)) for i in range(m)),
            )
            return GFOArray._from_raw(result, (m,))

        if self._ndim == 1 and other._ndim == 2:
            k, n = other._shape
            sd, od = self._data, other._data
            result = _array_mod.array(
                _DOUBLE,
                (_sum(sd[i] * od[i * n + j] for i in range(k)) for j in range(n)),
            )
            return GFOArray._from_raw(result, (n,))

        if self._ndim == 2 and other._ndim == 2:
            m, k1 = self._shape
            k2, n = other._shape
            if k1 != k2:
                raise ValueError(
                    f"Incompatible dimensions: {self._shape} @ {other._shape}"
                )
            sd, od = self._data, other._data
            result = _array_mod.array(_DOUBLE, [0.0] * (m * n))
            for i in range(m):
                si = i * k1
                ri = i * n
                for j in range(n):
                    s = 0.0
                    for k in range(k1):
                        s += sd[si + k] * od[k * n + j]
                    result[ri + j] = s
            return GFOArray._from_raw(result, (m, n))

        raise ValueError(f"Cannot matmul shapes {self._shape} and {other._shape}")

    def __rmatmul__(self, other):
        return GFOArray(other).__matmul__(self)

    def __eq__(self, other):
        return self._cmpop(other, _eq)

    def __ne__(self, other):
        return self._cmpop(other, _ne)

    def __lt__(self, other):
        return self._cmpop(other, _lt)

    def __le__(self, other):
        return self._cmpop(other, _le)

    def __gt__(self, other):
        return self._cmpop(other, _gt)

    def __ge__(self, other):
        return self._cmpop(other, _ge)

    def tolist(self):
        if self._ndim == 1:
            return list(self._data)
        ncols = self._shape[1]
        return [
            list(self._data[i * ncols : (i + 1) * ncols]) for i in range(self._shape[0])
        ]

    def flatten(self):
        n = len(self._data)
        if isinstance(self._data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, self._data), (n,))
        return GFOArray._from_raw(list(self._data), (n,))

    def ravel(self):
        return self.flatten()

    def reshape(self, *args):
        if len(args) == 1:
            shape = args[0]
        else:
            shape = args
        if isinstance(shape, int):
            shape = (shape,)
        neg_idx = None
        known = 1
        for i, d in enumerate(shape):
            if d == -1:
                neg_idx = i
            else:
                known *= d
        if neg_idx is not None:
            inferred = len(self._data) // known
            shape = shape[:neg_idx] + (inferred,) + shape[neg_idx + 1 :]
        if isinstance(self._data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, self._data), shape)
        return GFOArray._from_raw(list(self._data), shape)

    def transpose(self):
        if self._ndim == 1:
            return self.copy()
        rows, cols = self._shape
        sd = self._data
        if isinstance(sd, _array_mod.array):
            result = _array_mod.array(_DOUBLE, [0.0] * len(sd))
        else:
            result = [None] * len(sd)
        for r in range(rows):
            for c in range(cols):
                result[c * rows + r] = sd[r * cols + c]
        return GFOArray._from_raw(result, (cols, rows))

    def copy(self):
        if isinstance(self._data, _array_mod.array):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, self._data), self._shape
            )
        return GFOArray._from_raw(list(self._data), self._shape)

    def astype(self, dtype):
        result = self.copy()
        result._apply_dtype(dtype)
        return result

    def sum(self, axis=None):
        if axis is None:
            return _sum(self._data)
        if self._ndim == 1:
            return _sum(self._data)
        ncols = self._shape[1]
        nrows = self._shape[0]
        sd = self._data
        if axis == 0:
            if isinstance(sd, _array_mod.array):
                result = _array_mod.array(_DOUBLE, [0.0] * ncols)
            else:
                result = [0.0] * ncols
            for r in range(nrows):
                base = r * ncols
                for c in range(ncols):
                    result[c] += sd[base + c]
            return GFOArray._from_raw(result, (ncols,))
        if axis == 1:
            vals = []
            for r in range(nrows):
                base = r * ncols
                vals.append(_sum(sd[base : base + ncols]))
            if isinstance(sd, _array_mod.array):
                return GFOArray._from_raw(_array_mod.array(_DOUBLE, vals), (nrows,))
            return GFOArray._from_raw(vals, (nrows,))
        raise ValueError(f"Invalid axis {axis}")

    def mean(self, axis=None):
        if axis is None:
            n = len(self._data)
            return _sum(self._data) / n if n else 0.0
        s = self.sum(axis=axis)
        n = self._shape[axis]
        if isinstance(s, GFOArray):
            if isinstance(s._data, _array_mod.array):
                return GFOArray._from_raw(
                    _array_mod.array(_DOUBLE, (x / n for x in s._data)),
                    s._shape,
                )
            return GFOArray._from_raw([x / n for x in s._data], s._shape)
        return s / n

    def std(self, axis=None, ddof=0):
        m = self.mean(axis=axis)
        if axis is None:
            n = len(self._data)
            variance = _sum((x - m) ** 2 for x in self._data) / (n - ddof)
            return _m_sqrt(variance)
        raise NotImplementedError("Axis-aware std not implemented")

    def var(self, axis=None, ddof=0):
        return self.std(axis=axis, ddof=ddof) ** 2

    def min(self, axis=None):
        if axis is None:
            return _min(self._data)
        if self._ndim != 2:
            raise NotImplementedError("Axis-aware min requires 2D array")
        nrows, ncols = self._shape
        data = self._data
        if axis == 1:
            result = [_min(data[r * ncols : (r + 1) * ncols]) for r in range(nrows)]
        else:
            result = [
                _min(data[r * ncols + c] for r in range(nrows)) for c in range(ncols)
            ]
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))

    def max(self, axis=None):
        if axis is None:
            return _max(self._data)
        if self._ndim != 2:
            raise NotImplementedError("Axis-aware max requires 2D array")
        nrows, ncols = self._shape
        data = self._data
        if axis == 1:
            result = [_max(data[r * ncols : (r + 1) * ncols]) for r in range(nrows)]
        else:
            result = [
                _max(data[r * ncols + c] for r in range(nrows)) for c in range(ncols)
            ]
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))

    def argmax(self, axis=None):
        if axis is None:
            data = self._data
            best_val = data[0]
            best_i = 0
            for i in range(1, len(data)):
                if data[i] > best_val:
                    best_val = data[i]
                    best_i = i
            return best_i
        raise NotImplementedError("Axis-aware argmax not implemented")

    def argmin(self, axis=None):
        if axis is None:
            data = self._data
            best_val = data[0]
            best_i = 0
            for i in range(1, len(data)):
                if data[i] < best_val:
                    best_val = data[i]
                    best_i = i
            return best_i
        raise NotImplementedError("Axis-aware argmin not implemented")

    def argsort(self, axis=-1):
        data = self._data
        indices = sorted(range(len(data)), key=data.__getitem__)
        return GFOArray._from_raw(indices, (len(indices),))

    def any(self):
        return _any(self._data)

    def all(self):
        return _all(self._data)


_eq = operator.eq
_ne = operator.ne
_lt = operator.lt
_le = operator.le
_gt = operator.gt
_ge = operator.ge


def _apply_unary(x, func):
    if isinstance(x, GFOArray):
        if isinstance(x._data, _array_mod.array):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, map(func, x._data)), x._shape
            )
        return GFOArray._from_raw(list(map(func, x._data)), x._shape)
    return func(x)


def array(data, dtype=None) -> GFOArray:
    return GFOArray(data, dtype=dtype)


def asarray(data, dtype=None) -> GFOArray:
    if isinstance(data, GFOArray):
        return data if dtype is None else data.astype(dtype)
    return GFOArray(data, dtype=dtype)


def zeros(shape: Shape, dtype=float) -> GFOArray:
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for d in shape:
        total *= d
    if dtype is bool:
        return GFOArray._from_raw([False] * total, shape)
    if dtype is object:
        return GFOArray._from_raw([0] * total, shape)
    return GFOArray._from_raw(_array_mod.array(_DOUBLE, [0.0] * total), shape)


def zeros_like(a, dtype=None) -> GFOArray:
    if isinstance(a, GFOArray):
        sh = a.shape
    elif hasattr(a, "shape"):
        sh = a.shape
    else:
        sh = (len(a),)
    return zeros(sh, dtype if dtype is not None else float)


def ones(shape: Shape, dtype=float) -> GFOArray:
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for d in shape:
        total *= d
    if dtype is bool:
        return GFOArray._from_raw([True] * total, shape)
    return GFOArray._from_raw(_array_mod.array(_DOUBLE, [1.0] * total), shape)


def empty(shape: Shape, dtype=float) -> GFOArray:
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for d in shape:
        total *= d
    if dtype is object:
        return GFOArray._from_raw([None] * total, shape)
    if dtype is bool:
        return GFOArray._from_raw([False] * total, shape)
    return GFOArray._from_raw(_array_mod.array(_DOUBLE, [0.0] * total), shape)


def empty_like(a, dtype=None) -> GFOArray:
    return zeros_like(a, dtype=dtype)


def full(shape: Shape, fill_value, dtype=None) -> GFOArray:
    if isinstance(shape, int):
        shape = (shape,)
    val = dtype(fill_value) if dtype else fill_value
    total = 1
    for d in shape:
        total *= d
    if isinstance(val, bool):
        return GFOArray._from_raw([val] * total, shape)
    try:
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, [val] * total), shape)
    except TypeError:
        return GFOArray._from_raw([val] * total, shape)


def arange(start, stop=None, step=1, dtype=None) -> GFOArray:
    if stop is None:
        start, stop = 0, start
    result = []
    val = start
    while val < stop:
        result.append(val)
        val += step
    if dtype:
        result = [dtype(x) for x in result]
        return GFOArray._from_raw(result, (len(result),))
    return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))


def linspace(start, stop, num=50, endpoint=True, dtype=None) -> GFOArray:
    if num < 0:
        raise ValueError("Number of samples must be non-negative")
    if num == 0:
        return GFOArray._from_raw(_array_mod.array(_DOUBLE), (0,))
    if num == 1:
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, [float(start)]), (1,))
    step = (stop - start) / (num - 1 if endpoint else num)
    result = _array_mod.array(_DOUBLE, (start + i * step for i in range(num)))
    if dtype:
        result = _array_mod.array(_DOUBLE, (dtype(x) for x in result))
    return GFOArray._from_raw(result, (num,))


def eye(n: int, m: int = None, dtype=float) -> GFOArray:
    if m is None:
        m = n
    total = n * m
    result = _array_mod.array(_DOUBLE, [0.0] * total)
    for i in range(min(n, m)):
        result[i * m + i] = 1.0
    return GFOArray._from_raw(result, (n, m))


def diag(v, k=0):
    if isinstance(v, GFOArray):
        if v._ndim == 2:
            rows, cols = v._shape
            result = []
            if k >= 0:
                for i in range(min(rows, cols - k)):
                    result.append(v._data[i * cols + i + k])
            else:
                for i in range(min(rows + k, cols)):
                    result.append(v._data[(i - k) * cols + i])
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))
        else:
            n = len(v._data) + _abs(k)
            total = n * n
            result = _array_mod.array(_DOUBLE, [0.0] * total)
            if k >= 0:
                for i, val in enumerate(v._data):
                    result[i * n + i + k] = val
            else:
                for i, val in enumerate(v._data):
                    result[(i - k) * n + i] = val
            return GFOArray._from_raw(result, (n, n))
    return diag(GFOArray(v), k)


def meshgrid(*arrays, indexing="xy"):
    if len(arrays) != 2:
        raise NotImplementedError("meshgrid only supports 2D")

    x = list(arrays[0]) if isinstance(arrays[0], GFOArray) else list(arrays[0])
    y = list(arrays[1]) if isinstance(arrays[1], GFOArray) else list(arrays[1])

    if indexing == "xy":
        xx_data = []
        yy_data = []
        for yi in y:
            xx_data.extend(x)
            yy_data.extend([yi] * len(x))
        shape = (len(y), len(x))
    else:
        xx_data = []
        yy_data = []
        for xi in x:
            xx_data.extend([xi] * len(y))
            yy_data.extend(y)
        shape = (len(x), len(y))

    return (
        GFOArray._from_raw(_array_mod.array(_DOUBLE, xx_data), shape),
        GFOArray._from_raw(_array_mod.array(_DOUBLE, yy_data), shape),
    )


def ndim(x) -> int:
    if isinstance(x, GFOArray):
        return x.ndim
    if isinstance(x, list | tuple):
        if len(x) > 0 and isinstance(x[0], list | tuple):
            return 2
        return 1
    return 0


def shape(x) -> tuple[int, ...]:
    if isinstance(x, GFOArray):
        return x.shape
    if isinstance(x, list | tuple):
        if len(x) > 0 and isinstance(x[0], list | tuple):
            return (len(x), len(x[0]))
        return (len(x),)
    return ()


def exp(x):
    return _apply_unary(x, _m_exp)


def log(x):
    return _apply_unary(x, _m_log)


def log10(x):
    return _apply_unary(x, _m_log10)


def sqrt(x):
    return _apply_unary(x, _m_sqrt)


def abs(x):
    return _apply_unary(x, _abs)


def power(x, p):
    if isinstance(x, GFOArray):
        return x._binop(p, _pow_op)
    return x**p


def square(x):
    return power(x, 2)


def sin(x):
    return _apply_unary(x, _m_sin)


def cos(x):
    return _apply_unary(x, _m_cos)


def clip(x, a_min, a_max):
    if isinstance(x, GFOArray):
        data = x._data
        min_is_arr = isinstance(a_min, GFOArray)
        max_is_arr = isinstance(a_max, GFOArray)
        if min_is_arr and max_is_arr:
            result = [
                _max(lo, _min(v, hi))
                for v, lo, hi in zip(data, a_min._data, a_max._data)
            ]
        elif max_is_arr:
            result = [_max(a_min, _min(v, m)) for v, m in zip(data, a_max._data)]
        elif min_is_arr:
            result = [_max(lo, _min(v, a_max)) for v, lo in zip(data, a_min._data)]
        else:
            result = [_max(a_min, _min(v, a_max)) for v in data]
        if isinstance(data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), x._shape)
        return GFOArray._from_raw(result, x._shape)
    return _max(a_min, _min(x, a_max))


def rint(x):
    if isinstance(x, GFOArray):
        data = x._data
        result = [_round(v) for v in data]
        if isinstance(data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), x._shape)
        return GFOArray._from_raw(result, x._shape)
    return _round(x)


def round(x, decimals=0):
    if isinstance(x, GFOArray):
        data = x._data
        result = [_round(v, decimals) for v in data]
        if isinstance(data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), x._shape)
        return GFOArray._from_raw(result, x._shape)
    return _round(x, decimals)


def floor(x):
    return _apply_unary(x, _m_floor)


def ceil(x):
    return _apply_unary(x, _m_ceil)


def maximum(x, y):
    if isinstance(x, GFOArray):
        if isinstance(y, GFOArray):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, map(_max, x._data, y._data))
                if isinstance(x._data, _array_mod.array)
                and isinstance(y._data, _array_mod.array)
                else list(map(_max, x._data, y._data)),
                x._shape,
            )
        result = list(map(_max, x._data, _repeat(y)))
        if isinstance(x._data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), x._shape)
        return GFOArray._from_raw(result, x._shape)
    if isinstance(y, GFOArray):
        result = list(map(_max, _repeat(x), y._data))
        if isinstance(y._data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), y._shape)
        return GFOArray._from_raw(result, y._shape)
    return _max(x, y)


def minimum(x, y):
    if isinstance(x, GFOArray):
        if isinstance(y, GFOArray):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, map(_min, x._data, y._data))
                if isinstance(x._data, _array_mod.array)
                and isinstance(y._data, _array_mod.array)
                else list(map(_min, x._data, y._data)),
                x._shape,
            )
        result = list(map(_min, x._data, _repeat(y)))
        if isinstance(x._data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), x._shape)
        return GFOArray._from_raw(result, x._shape)
    if isinstance(y, GFOArray):
        result = list(map(_min, _repeat(x), y._data))
        if isinstance(y._data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), y._shape)
        return GFOArray._from_raw(result, y._shape)
    return _min(x, y)


def greater(x, y):
    return GFOArray(x) > y


def less(x, y):
    return GFOArray(x) < y


def equal(x, y):
    return GFOArray(x) == y


def isnan(x):
    if isinstance(x, GFOArray):
        return GFOArray._from_raw(list(map(_m_isnan, x._data)), x._shape)
    return _m_isnan(x)


def isinf(x):
    if isinstance(x, GFOArray):
        return GFOArray._from_raw(list(map(_m_isinf, x._data)), x._shape)
    return _m_isinf(x)


def isfinite(x):
    if isinstance(x, GFOArray):
        return GFOArray._from_raw(list(map(_m_isfinite, x._data)), x._shape)
    return _m_isfinite(x)


def sum(x, axis=None):
    if isinstance(x, GFOArray):
        return x.sum(axis=axis)
    return _sum(x)


def mean(x, axis=None):
    if isinstance(x, GFOArray):
        return x.mean(axis=axis)
    return _sum(x) / len(x)


def std(x, axis=None, ddof=0):
    if isinstance(x, GFOArray):
        return x.std(axis=axis, ddof=ddof)
    m = mean(x)
    return _m_sqrt(_sum((v - m) ** 2 for v in x) / (len(x) - ddof))


def var(x, axis=None, ddof=0):
    return std(x, axis=axis, ddof=ddof) ** 2


def prod(x, axis=None):
    if isinstance(x, GFOArray):
        data = x._data
    else:
        data = x
    result = 1
    for v in data:
        result *= v
    return result


def cumsum(x, axis=None):
    if isinstance(x, GFOArray):
        data = x._data
    else:
        data = x
    result = []
    total = 0
    for v in data:
        total += v
        result.append(total)
    return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))


def argmax(x, axis=None):
    if isinstance(x, GFOArray):
        return x.argmax(axis=axis)
    xl = list(x)
    return xl.index(_max(xl))


def argmin(x, axis=None):
    if isinstance(x, GFOArray):
        return x.argmin(axis=axis)
    xl = list(x)
    return xl.index(_min(xl))


def argsort(x, axis=-1):
    if isinstance(x, GFOArray):
        data = x._data
    else:
        data = list(x)
    indices = sorted(range(len(data)), key=data.__getitem__)
    return GFOArray._from_raw(indices, (len(indices),))


def where(condition, x=None, y=None):
    if isinstance(condition, GFOArray):
        cond_data = condition._data
    else:
        cond_data = list(condition)

    if x is None and y is None:
        indices = [i for i, c in enumerate(cond_data) if c]
        return (GFOArray._from_raw(indices, (len(indices),)),)

    if isinstance(x, GFOArray):
        x_data = x._data
    elif isinstance(x, list | tuple):
        x_data = x
    else:
        x_data = None

    if isinstance(y, GFOArray):
        y_data = y._data
    elif isinstance(y, list | tuple):
        y_data = y
    else:
        y_data = None

    n = len(cond_data)
    result = []
    for i in range(n):
        if cond_data[i]:
            result.append(x_data[i] if x_data is not None else x)
        else:
            result.append(y_data[i] if y_data is not None else y)

    try:
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (n,))
    except TypeError:
        return GFOArray._from_raw(result, (n,))


def nonzero(x):
    if isinstance(x, GFOArray):
        data = x._data
    else:
        data = list(x)
    indices = [i for i, v in enumerate(data) if v]
    return (GFOArray._from_raw(indices, (len(indices),)),)


def _binary_search(arr, val, side):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if (side == "left" and arr[mid] < val) or (side == "right" and arr[mid] <= val):
            lo = mid + 1
        else:
            hi = mid
    return lo


def searchsorted(a, v, side="left"):
    if isinstance(a, GFOArray):
        a_data = a._data
    else:
        a_data = list(a)

    if isinstance(v, GFOArray | list | tuple):
        v_list = list(v) if not isinstance(v, GFOArray) else list(v._data)
        results = [_binary_search(a_data, val, side) for val in v_list]
        return GFOArray._from_raw(
            results,
            (len(results),),
        )
    return _binary_search(a_data, v, side)


def take(a, indices, axis=None):
    if isinstance(a, GFOArray):
        data = a._data
    else:
        data = list(a)

    if isinstance(indices, GFOArray):
        idx = [int(i) for i in indices._data]
    elif isinstance(indices, list | tuple):
        idx = [int(i) for i in indices]
    else:
        return data[int(indices)]

    result = [data[i] for i in idx]
    if isinstance(data, _array_mod.array):
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))
    return GFOArray._from_raw(result, (len(result),))


def unique(x, return_index=False, return_inverse=False, return_counts=False):
    if isinstance(x, GFOArray):
        flat = list(x._data)
    else:
        flat = list(x)

    seen = {}
    for i, v in enumerate(flat):
        if v not in seen:
            seen[v] = i

    result = sorted(seen.keys())

    if not any([return_index, return_inverse, return_counts]):
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))

    ret = [GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))]
    if return_index:
        indices = [seen[v] for v in result]
        ret.append(
            GFOArray._from_raw(
                _array_mod.array(_DOUBLE, (float(i) for i in indices)),
                (len(indices),),
            )
        )
    if return_inverse:
        result_idx = {v: i for i, v in enumerate(result)}
        inverse = [result_idx[v] for v in flat]
        ret.append(
            GFOArray._from_raw(
                _array_mod.array(_DOUBLE, (float(i) for i in inverse)),
                (len(inverse),),
            )
        )
    if return_counts:
        counts = [flat.count(v) for v in result]
        ret.append(
            GFOArray._from_raw(
                _array_mod.array(_DOUBLE, (float(c) for c in counts)),
                (len(counts),),
            )
        )
    return tuple(ret)


def intersect1d(ar1, ar2):
    if isinstance(ar1, GFOArray):
        set1 = set(ar1._data)
    else:
        set1 = set(ar1)
    if isinstance(ar2, GFOArray):
        set2 = set(ar2._data)
    else:
        set2 = set(ar2)
    result = sorted(set1 & set2)
    return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))


def isin(element, test_elements):
    if isinstance(test_elements, GFOArray):
        test_set = set(test_elements._data)
    else:
        test_set = set(test_elements)
    if isinstance(element, GFOArray):
        return GFOArray._from_raw(
            [v in test_set for v in element._data], element._shape
        )
    return element in test_set


def reshape(x, shape):
    if isinstance(x, GFOArray):
        return x.reshape(shape)
    return GFOArray(x).reshape(shape)


def transpose(x, axes=None):
    if isinstance(x, GFOArray):
        return x.transpose()
    return GFOArray(x).transpose()


def ravel(x):
    if isinstance(x, GFOArray):
        return x.ravel()
    return GFOArray(x).ravel()


def flatten(x):
    if isinstance(x, GFOArray):
        return x.flatten()
    return GFOArray(x).flatten()


def concatenate(arrays, axis=0):
    if axis == 0:
        first = arrays[0]
        if isinstance(first, GFOArray) and first._ndim == 2:
            ncols = first._shape[1]
            all_data = _array_mod.array(_DOUBLE)
            total_rows = 0
            for arr in arrays:
                if isinstance(arr, GFOArray):
                    all_data.extend(arr._data)
                    total_rows += arr._shape[0]
                else:
                    for row in arr:
                        all_data.extend(row)
                    total_rows += len(arr)
            return GFOArray._from_raw(all_data, (total_rows, ncols))

    result = _array_mod.array(_DOUBLE)
    for arr in arrays:
        if isinstance(arr, GFOArray):
            if isinstance(arr._data, _array_mod.array):
                result.extend(arr._data)
            else:
                result.extend(_array_mod.array(_DOUBLE, arr._data))
        else:
            result.extend(_array_mod.array(_DOUBLE, arr))
    return GFOArray._from_raw(result, (len(result),))


def stack(arrays, axis=0):
    arr_list = []
    for a in arrays:
        if isinstance(a, GFOArray):
            arr_list.append(list(a._data))
        else:
            arr_list.append(list(a))
    nrows = len(arr_list)
    ncols = len(arr_list[0]) if arr_list else 0
    flat = _array_mod.array(_DOUBLE)
    for row in arr_list:
        flat.extend(row)
    return GFOArray._from_raw(flat, (nrows, ncols))


def vstack(arrays):
    return stack(arrays, axis=0)


def hstack(arrays):
    result = _array_mod.array(_DOUBLE)
    for arr in arrays:
        if isinstance(arr, GFOArray):
            if isinstance(arr._data, _array_mod.array):
                result.extend(arr._data)
            else:
                result.extend(arr._data)
        else:
            result.extend(arr)
    return GFOArray._from_raw(result, (len(result),))


def tile(x, reps):
    if isinstance(x, GFOArray):
        data = x._data
    else:
        data = list(x)
    if isinstance(reps, int):
        if isinstance(data, _array_mod.array):
            result = _array_mod.array(_DOUBLE)
            for _ in range(reps):
                result.extend(data)
            return GFOArray._from_raw(result, (len(result),))
        return GFOArray._from_raw(list(data) * reps, (len(data) * reps,))
    raise NotImplementedError("Multi-dimensional tile not supported")


def repeat(x, repeats, axis=None):
    if isinstance(x, GFOArray):
        data = x._data
    else:
        data = list(x)
    if isinstance(repeats, int):
        result = []
        for v in data:
            result.extend([v] * repeats)
        n = len(result)
        if isinstance(data, _array_mod.array):
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (n,))
        return GFOArray._from_raw(result, (n,))
    raise NotImplementedError("Variable repeats not supported")


def array_split(x, indices_or_sections, axis=0):
    if isinstance(x, GFOArray):
        data = x._data
    else:
        data = list(x)

    n = len(data)
    if isinstance(indices_or_sections, int):
        n_sec = indices_or_sections
        sec_size = n // n_sec
        remainder = n % n_sec
        result = []
        start = 0
        for i in range(n_sec):
            end = start + sec_size + (1 if i < remainder else 0)
            chunk = data[start:end]
            if isinstance(data, _array_mod.array):
                result.append(GFOArray._from_raw(chunk, (len(chunk),)))
            else:
                result.append(GFOArray._from_raw(list(chunk), (len(chunk),)))
            start = end
        return result
    else:
        result = []
        prev = 0
        for idx in indices_or_sections:
            chunk = data[prev:idx]
            if isinstance(data, _array_mod.array):
                result.append(GFOArray._from_raw(chunk, (len(chunk),)))
            else:
                result.append(GFOArray._from_raw(list(chunk), (len(chunk),)))
            prev = idx
        chunk = data[prev:]
        if isinstance(data, _array_mod.array):
            result.append(GFOArray._from_raw(chunk, (len(chunk),)))
        else:
            result.append(GFOArray._from_raw(list(chunk), (len(chunk),)))
        return result


def split(x, indices_or_sections, axis=0):
    return array_split(x, indices_or_sections, axis)


def dot(a, b):
    if isinstance(a, GFOArray):
        a_data = a._data
    else:
        a_data = list(a)
    if isinstance(b, GFOArray):
        b_data = b._data
    else:
        b_data = list(b)
    if len(a_data) == len(b_data):
        return _sum(map(_mul, a_data, b_data))
    raise NotImplementedError("Matrix dot product not fully implemented")


def matmul(a, b):
    a_arr = a if isinstance(a, GFOArray) else GFOArray(a)
    b_arr = b if isinstance(b, GFOArray) else GFOArray(b)
    return a_arr.__matmul__(b_arr)


def outer(a, b):
    if isinstance(a, GFOArray):
        a_data = a._data
    else:
        a_data = list(a)
    if isinstance(b, GFOArray):
        b_data = b._data
    else:
        b_data = list(b)
    m, n = len(a_data), len(b_data)
    result = _array_mod.array(_DOUBLE, [0.0] * (m * n))
    for i in range(m):
        ai = a_data[i]
        base = i * n
        for j in range(n):
            result[base + j] = ai * b_data[j]
    return GFOArray._from_raw(result, (m, n))


class linalg:
    """Linear algebra namespace."""

    class LinAlgError(Exception):
        pass

    @staticmethod
    def solve(a, b):
        raise NotImplementedError("linalg.solve requires numpy/scipy.")

    @staticmethod
    def lstsq(a, b, rcond=None):
        raise NotImplementedError("linalg.lstsq requires numpy/scipy.")

    @staticmethod
    def pinv(a):
        raise NotImplementedError("linalg.pinv requires numpy/scipy.")

    @staticmethod
    def inv(a):
        raise NotImplementedError("linalg.inv requires numpy/scipy.")

    @staticmethod
    def eigvalsh(a):
        raise NotImplementedError("linalg.eigvalsh requires numpy/scipy.")

    @staticmethod
    def eigh(a):
        raise NotImplementedError("linalg.eigh requires numpy/scipy.")

    @staticmethod
    def norm(x, ord=None, axis=None):
        if isinstance(x, GFOArray):
            data = x._data
        else:
            data = list(x)
        if ord is None or ord == 2:
            return _m_sqrt(_sum(v * v for v in data))
        if ord == 1:
            return _sum(map(_abs, data))
        if ord == inf:
            return _max(map(_abs, data))
        raise NotImplementedError(f"Norm order {ord} not implemented")

    @staticmethod
    def det(a):
        raise NotImplementedError("linalg.det requires numpy/scipy.")


def triu(m, k=0):
    """Upper triangle of a matrix."""
    if not isinstance(m, GFOArray):
        m = GFOArray(m)
    if m._ndim != 2:
        raise ValueError("triu requires 2D array")
    rows, cols = m._shape
    result = _array_mod.array(_DOUBLE, [0.0] * (rows * cols))
    sd = m._data
    for r in range(rows):
        for c in range(cols):
            if c >= r + k:
                result[r * cols + c] = sd[r * cols + c]
    return GFOArray._from_raw(result, (rows, cols))


class random:
    """Random number generation namespace."""

    _rng = py_random.Random()

    @staticmethod
    def seed(seed=None):
        random._rng = py_random.Random(seed)

    @staticmethod
    def default_rng(seed=None):
        return _Generator(seed)

    @staticmethod
    def randint(low, high=None, size=None):
        if high is None:
            low, high = 0, low
        rng = random._rng
        if size is None:
            return rng.randint(low, high - 1)
        if isinstance(size, int):
            return GFOArray._from_raw(
                [rng.randint(low, high - 1) for _ in range(size)],
                (size,),
            )
        total = 1
        for s in size:
            total *= s
        data = [rng.randint(low, high - 1) for _ in range(total)]
        return GFOArray._from_raw(data, (total,)).reshape(size)

    @staticmethod
    def choice(a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a = list(range(a))
        elif isinstance(a, GFOArray):
            a = list(a._data)
        else:
            a = list(a)
        rng = random._rng
        if size is None:
            if p is not None:
                return rng.choices(a, weights=p, k=1)[0]
            return rng.choice(a)
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        if replace:
            if p is not None:
                result = rng.choices(a, weights=p, k=n)
            else:
                result = [rng.choice(a) for _ in range(n)]
        else:
            result = rng.sample(a, n)
        try:
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (n,))
        except TypeError:
            return GFOArray._from_raw(result, (n,))

    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        rng = random._rng
        if size is None:
            return rng.uniform(low, high)
        if isinstance(size, int):
            return GFOArray._from_raw(
                _array_mod.array(
                    _DOUBLE, (rng.uniform(low, high) for _ in range(size))
                ),
                (size,),
            )
        total = 1
        for s in size:
            total *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (rng.uniform(low, high) for _ in range(total))),
            (total,),
        ).reshape(size)

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        rng = random._rng
        if size is None:
            return rng.gauss(loc, scale)
        if isinstance(size, int):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, (rng.gauss(loc, scale) for _ in range(size))),
                (size,),
            )
        total = 1
        for s in size:
            total *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (rng.gauss(loc, scale) for _ in range(total))),
            (total,),
        ).reshape(size)

    @staticmethod
    def random_sample(size=None):
        rng = random._rng
        if size is None:
            return rng.random()
        if isinstance(size, int):
            return GFOArray._from_raw(
                _array_mod.array(_DOUBLE, (rng.random() for _ in range(size))),
                (size,),
            )
        total = 1
        for s in size:
            total *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (rng.random() for _ in range(total))),
            (total,),
        ).reshape(size)

    @staticmethod
    def laplace(loc=0.0, scale=1.0, size=None):
        rng = random._rng

        def _sample():
            u = rng.random() - 0.5
            return loc - scale * (1 if u >= 0 else -1) * _m_log(1 - 2 * _abs(u))

        if size is None:
            return _sample()
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (_sample() for _ in range(n))),
            (n,),
        )

    @staticmethod
    def logistic(loc=0.0, scale=1.0, size=None):
        rng = random._rng

        def _sample():
            u = rng.random()
            return loc + scale * _m_log(u / (1 - u))

        if size is None:
            return _sample()
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (_sample() for _ in range(n))),
            (n,),
        )

    @staticmethod
    def gumbel(loc=0.0, scale=1.0, size=None):
        rng = random._rng

        def _sample():
            u = rng.random()
            return loc - scale * _m_log(-_m_log(u))

        if size is None:
            return _sample()
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (_sample() for _ in range(n))),
            (n,),
        )

    @staticmethod
    def shuffle(x):
        if isinstance(x, GFOArray):
            if isinstance(x._data, _array_mod.array):
                temp = list(x._data)
                random._rng.shuffle(temp)
                x._data = _array_mod.array(_DOUBLE, temp)
            else:
                random._rng.shuffle(x._data)
        else:
            random._rng.shuffle(x)

    @staticmethod
    def permutation(x):
        if isinstance(x, int):
            result = list(range(x))
        elif isinstance(x, GFOArray):
            result = list(x._data)
        else:
            result = list(x)
        random._rng.shuffle(result)
        return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (len(result),))

    class RandomState:
        def __init__(self, seed=None):
            self._rng = py_random.Random(seed)

        def randint(self, low, high=None, size=None):
            if high is None:
                low, high = 0, low
            if size is None:
                return self._rng.randint(low, high - 1)
            return GFOArray._from_raw(
                [self._rng.randint(low, high - 1) for _ in range(size)],
                (size,),
            )

        def choice(self, a, size=None, replace=True):
            if isinstance(a, int):
                a = list(range(a))
            if size is None:
                return self._rng.choice(a)
            if replace:
                return GFOArray._from_raw(
                    [self._rng.choice(a) for _ in range(size)], (size,)
                )
            return GFOArray._from_raw(self._rng.sample(a, size), (size,))

        def uniform(self, low=0.0, high=1.0, size=None):
            if size is None:
                return self._rng.uniform(low, high)
            return GFOArray._from_raw(
                _array_mod.array(
                    _DOUBLE,
                    (self._rng.uniform(low, high) for _ in range(size)),
                ),
                (size,),
            )

        def normal(self, loc=0.0, scale=1.0, size=None):
            if size is None:
                return self._rng.gauss(loc, scale)
            return GFOArray._from_raw(
                _array_mod.array(
                    _DOUBLE,
                    (self._rng.gauss(loc, scale) for _ in range(size)),
                ),
                (size,),
            )


class _Generator:
    """numpy.random.Generator-compatible RNG for the pure backend."""

    def __init__(self, seed=None):
        if isinstance(seed, int) or seed is None:
            self._rng = py_random.Random(seed)
        else:
            self._rng = py_random.Random(int(seed))

    def random(self, size=None):
        rng = self._rng
        if size is None:
            return rng.random()
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (rng.random() for _ in range(n))),
            (n,),
        )

    def standard_normal(self, size=None):
        rng = self._rng
        if size is None:
            return rng.gauss(0.0, 1.0)
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (rng.gauss(0.0, 1.0) for _ in range(n))),
            (n,),
        )

    def uniform(self, low=0.0, high=1.0, size=None):
        rng = self._rng
        if size is None:
            return rng.uniform(low, high)
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (rng.uniform(low, high) for _ in range(n))),
            (n,),
        )

    def integers(self, low, high=None, size=None, endpoint=False):
        if high is None:
            low, high = 0, low
        if not endpoint:
            high_val = high - 1
        else:
            high_val = high
        rng = self._rng
        low_arr = isinstance(low, GFOArray)
        high_arr = isinstance(high_val, GFOArray)
        if low_arr or high_arr:
            n = len(low) if low_arr else len(high_val)
            lows = [int(v) for v in low._data] if low_arr else [int(low)] * n
            highs = (
                [int(v) for v in high_val._data] if high_arr else [int(high_val)] * n
            )
            return GFOArray._from_raw(
                [float(rng.randint(l, h)) for l, h in zip(lows, highs)], (n,)
            )
        if size is None:
            return rng.randint(int(low), int(high_val))
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            [rng.randint(int(low), int(high_val)) for _ in range(n)], (n,)
        )

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a = list(range(a))
        elif isinstance(a, GFOArray):
            a = list(a._data)
        else:
            a = list(a)
        rng = self._rng
        if size is None:
            if p is not None:
                return rng.choices(
                    a, weights=list(p) if isinstance(p, GFOArray) else p, k=1
                )[0]
            return rng.choice(a)
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        if replace:
            if p is not None:
                weights = list(p) if isinstance(p, GFOArray) else p
                result = rng.choices(a, weights=weights, k=n)
            else:
                result = [rng.choice(a) for _ in range(n)]
        else:
            result = rng.sample(a, n)
        try:
            return GFOArray._from_raw(_array_mod.array(_DOUBLE, result), (n,))
        except TypeError:
            return GFOArray._from_raw(result, (n,))

    def normal(self, loc=0.0, scale=1.0, size=None):
        rng = self._rng
        loc_arr = isinstance(loc, GFOArray)
        scale_arr = isinstance(scale, GFOArray)
        if loc_arr or scale_arr:
            n = len(scale) if scale_arr else len(loc)
            locs = list(loc._data) if loc_arr else [loc] * n
            scales = list(scale._data) if scale_arr else [scale] * n
            return GFOArray._from_raw(
                _array_mod.array(
                    _DOUBLE, (rng.gauss(l, s) for l, s in zip(locs, scales))
                ),
                (n,),
            )
        if size is None:
            return rng.gauss(loc, scale)
        n = size if isinstance(size, int) else 1
        if not isinstance(size, int):
            for s in size:
                n *= s
        return GFOArray._from_raw(
            _array_mod.array(_DOUBLE, (rng.gauss(loc, scale) for _ in range(n))),
            (n,),
        )

    def shuffle(self, x):
        if isinstance(x, GFOArray):
            if isinstance(x._data, _array_mod.array):
                temp = list(x._data)
                self._rng.shuffle(temp)
                x._data = _array_mod.array(_DOUBLE, temp)
            else:
                self._rng.shuffle(x._data)
        else:
            self._rng.shuffle(x)


def copy(x):
    if isinstance(x, GFOArray):
        return x.copy()
    return GFOArray(x)


def allclose(a, b, rtol=1e-5, atol=1e-8):
    if isinstance(a, GFOArray):
        a_data = a._data
    else:
        a_data = a
    if isinstance(b, GFOArray):
        b_data = b._data
    else:
        b_data = b
    for x, y in zip(a_data, b_data):
        if _abs(x - y) > atol + rtol * _abs(y):
            return False
    return True


def all(x, axis=None):
    if isinstance(x, GFOArray):
        return _all(x._data)
    return _all(x)


def any(x, axis=None):
    if isinstance(x, GFOArray):
        return _any(x._data)
    return _any(x)


def invert(x):
    """Boolean inversion."""
    if isinstance(x, GFOArray):
        return GFOArray._from_raw([not v for v in x._data], x._shape)
    return not x

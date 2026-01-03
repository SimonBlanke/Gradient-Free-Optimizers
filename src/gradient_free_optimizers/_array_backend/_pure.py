"""
Pure Python backend - NumPy-compatible interface without dependencies.

This module provides pure Python implementations of array operations used by GFO.
It is significantly slower than NumPy but enables dependency-free operation.

Note: This is a minimal implementation covering only functions used by GFO.
Not all NumPy features are supported.
"""

import math
import random as py_random
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

# === Constants ===
inf = float("inf")
pi = math.pi
e = math.e
nan = float("nan")

# === Type Aliases ===
ArrayLike = Union["GFOArray", List, Tuple, int, float]
Shape = Union[int, Tuple[int, ...]]

# === Numeric Types (for compatibility) ===
int32 = int
int64 = int
float32 = float
float64 = float


class GFOArray:
    """
    Minimal array implementation for dependency-free operation.

    Supports 1D and 2D arrays with basic NumPy-like operations.
    """

    __slots__ = ("_data", "_shape", "_ndim")

    def __init__(self, data: Any, dtype: Optional[type] = None):
        if isinstance(data, GFOArray):
            self._data = data._data.copy()
            self._shape = data._shape
            self._ndim = data._ndim
        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                self._data = []
                self._shape = (0,)
                self._ndim = 1
            elif isinstance(data[0], (list, tuple)):
                # 2D array
                self._data = [list(row) for row in data]
                self._shape = (len(data), len(data[0]))
                self._ndim = 2
            else:
                # 1D array
                self._data = list(data)
                self._shape = (len(data),)
                self._ndim = 1
        else:
            # Scalar
            self._data = [data]
            self._shape = (1,)
            self._ndim = 1

        # Apply dtype conversion if specified
        if dtype is not None:
            self._apply_dtype(dtype)

    def _apply_dtype(self, dtype: type) -> None:
        """Apply dtype conversion to all elements."""
        if self._ndim == 1:
            self._data = [dtype(x) for x in self._data]
        else:
            self._data = [[dtype(x) for x in row] for row in self._data]

    def _get_flat(self) -> List:
        """Get flattened data."""
        if self._ndim == 1:
            return self._data
        return [x for row in self._data for x in row]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def size(self) -> int:
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    @property
    def dtype(self):
        """Return a simple dtype representation."""
        if self.size == 0:
            return float64
        sample = self._get_flat()[0] if self._get_flat() else 0
        return type(sample)

    def __len__(self) -> int:
        return self._shape[0]

    def __iter__(self) -> Iterator:
        if self._ndim == 1:
            return iter(self._data)
        return iter(GFOArray(row) for row in self._data)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if self._ndim == 2:
                row_idx, col_idx = idx
                if isinstance(row_idx, int) and isinstance(col_idx, int):
                    return self._data[row_idx][col_idx]
                # Handle slices
                if isinstance(row_idx, slice):
                    rows = self._data[row_idx]
                    if isinstance(col_idx, int):
                        return GFOArray([row[col_idx] for row in rows])
                    return GFOArray([row[col_idx] for row in rows])
        if isinstance(idx, int):
            if self._ndim == 1:
                return self._data[idx]
            return GFOArray(self._data[idx])
        if isinstance(idx, slice):
            if self._ndim == 1:
                return GFOArray(self._data[idx])
            return GFOArray(self._data[idx])
        if isinstance(idx, (list, GFOArray)):
            # Boolean or integer indexing
            idx_list = list(idx) if isinstance(idx, GFOArray) else idx
            if self._ndim == 1:
                if idx_list and isinstance(idx_list[0], bool):
                    return GFOArray([self._data[i] for i, b in enumerate(idx_list) if b])
                return GFOArray([self._data[i] for i in idx_list])
            elif self._ndim == 2:
                # 2D boolean indexing - select rows where mask is True
                if idx_list and isinstance(idx_list[0], bool):
                    return GFOArray([self._data[i] for i, b in enumerate(idx_list) if b])
                # Integer indexing for 2D
                return GFOArray([self._data[i] for i in idx_list])
        return self._data[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            if self._ndim == 1:
                self._data[idx] = value
            else:
                if isinstance(value, GFOArray):
                    self._data[idx] = value.tolist()
                else:
                    self._data[idx] = list(value)
        elif isinstance(idx, tuple) and self._ndim == 2:
            row_idx, col_idx = idx
            self._data[row_idx][col_idx] = value

    def __repr__(self) -> str:
        return f"GFOArray({self._data})"

    def __str__(self) -> str:
        return str(self._data)

    # === Arithmetic Operations ===

    def _apply_binary_op(self, other: Any, op: Callable) -> "GFOArray":
        """Apply binary operation element-wise."""
        if isinstance(other, GFOArray):
            other_flat = other._get_flat()
        elif isinstance(other, (list, tuple)):
            other_flat = list(other)
        else:
            # Scalar
            other_flat = None

        if other_flat is None:
            # Scalar operation
            if self._ndim == 1:
                result = [op(x, other) for x in self._data]
            else:
                result = [[op(x, other) for x in row] for row in self._data]
        else:
            # Element-wise operation
            flat = self._get_flat()
            result_flat = [op(a, b) for a, b in zip(flat, other_flat)]
            if self._ndim == 1:
                result = result_flat
            else:
                result = [result_flat[i:i+self._shape[1]]
                         for i in range(0, len(result_flat), self._shape[1])]
        return GFOArray(result)

    def __add__(self, other): return self._apply_binary_op(other, lambda a, b: a + b)
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self._apply_binary_op(other, lambda a, b: a - b)
    def __rsub__(self, other): return GFOArray(other)._apply_binary_op(self, lambda a, b: a - b)
    def __mul__(self, other): return self._apply_binary_op(other, lambda a, b: a * b)
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other): return self._apply_binary_op(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return GFOArray(other)._apply_binary_op(self, lambda a, b: a / b)
    def __floordiv__(self, other): return self._apply_binary_op(other, lambda a, b: a // b)
    def __pow__(self, other): return self._apply_binary_op(other, lambda a, b: a ** b)
    def __mod__(self, other): return self._apply_binary_op(other, lambda a, b: a % b)
    def __neg__(self): return GFOArray([-x for x in self._get_flat()])
    def __pos__(self): return GFOArray(self._data)
    def __abs__(self): return GFOArray([builtins_abs(x) for x in self._get_flat()])
    def __invert__(self):
        """Boolean negation (~arr) for boolean arrays."""
        if self._ndim == 1:
            return GFOArray([not x for x in self._data])
        return GFOArray([[not x for x in row] for row in self._data])

    # === Comparison Operations ===

    def __eq__(self, other): return self._apply_binary_op(other, lambda a, b: a == b)
    def __ne__(self, other): return self._apply_binary_op(other, lambda a, b: a != b)
    def __lt__(self, other): return self._apply_binary_op(other, lambda a, b: a < b)
    def __le__(self, other): return self._apply_binary_op(other, lambda a, b: a <= b)
    def __gt__(self, other): return self._apply_binary_op(other, lambda a, b: a > b)
    def __ge__(self, other): return self._apply_binary_op(other, lambda a, b: a >= b)

    # === Conversion Methods ===

    def tolist(self) -> List:
        if self._ndim == 1:
            return self._data.copy()
        return [row.copy() for row in self._data]

    def flatten(self) -> "GFOArray":
        return GFOArray(self._get_flat())

    def ravel(self) -> "GFOArray":
        return self.flatten()

    def reshape(self, shape: Shape) -> "GFOArray":
        flat = self._get_flat()
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) == 1:
            return GFOArray(flat)
        elif len(shape) == 2:
            rows, cols = shape
            result = [flat[i*cols:(i+1)*cols] for i in range(rows)]
            return GFOArray(result)
        raise ValueError(f"Reshape to {shape} not supported")

    def transpose(self) -> "GFOArray":
        if self._ndim == 1:
            return GFOArray(self._data)
        rows, cols = self._shape
        result = [[self._data[r][c] for r in range(rows)] for c in range(cols)]
        return GFOArray(result)

    @property
    def T(self) -> "GFOArray":
        return self.transpose()

    def copy(self) -> "GFOArray":
        return GFOArray(self._data)

    def astype(self, dtype: type) -> "GFOArray":
        result = GFOArray(self._data)
        result._apply_dtype(dtype)
        return result

    # === Aggregation Methods ===

    def sum(self, axis: Optional[int] = None):
        if axis is None:
            return builtins_sum(self._get_flat())
        if self._ndim == 1:
            return builtins_sum(self._data)
        if axis == 0:
            return GFOArray([builtins_sum(self._data[r][c] for r in range(self._shape[0]))
                           for c in range(self._shape[1])])
        if axis == 1:
            return GFOArray([builtins_sum(row) for row in self._data])
        raise ValueError(f"Invalid axis {axis}")

    def mean(self, axis: Optional[int] = None):
        if axis is None:
            flat = self._get_flat()
            return builtins_sum(flat) / len(flat) if flat else 0.0
        s = self.sum(axis=axis)
        n = self._shape[axis]
        if isinstance(s, GFOArray):
            return GFOArray([x / n for x in s._get_flat()])
        return s / n

    def std(self, axis: Optional[int] = None, ddof: int = 0):
        m = self.mean(axis=axis)
        if axis is None:
            flat = self._get_flat()
            variance = builtins_sum((x - m) ** 2 for x in flat) / (len(flat) - ddof)
            return math.sqrt(variance)
        # Simplified: axis-aware std not fully implemented
        raise NotImplementedError("Axis-aware std not implemented")

    def var(self, axis: Optional[int] = None, ddof: int = 0):
        std_val = self.std(axis=axis, ddof=ddof)
        return std_val ** 2

    def min(self, axis: Optional[int] = None):
        if axis is None:
            return builtins_min(self._get_flat())
        raise NotImplementedError("Axis-aware min not implemented")

    def max(self, axis: Optional[int] = None):
        if axis is None:
            return builtins_max(self._get_flat())
        raise NotImplementedError("Axis-aware max not implemented")

    def argmax(self, axis: Optional[int] = None):
        if axis is None:
            flat = self._get_flat()
            return flat.index(builtins_max(flat))
        raise NotImplementedError("Axis-aware argmax not implemented")

    def argmin(self, axis: Optional[int] = None):
        if axis is None:
            flat = self._get_flat()
            return flat.index(builtins_min(flat))
        raise NotImplementedError("Axis-aware argmin not implemented")


# Keep reference to builtins
import builtins
builtins_sum = builtins.sum
builtins_min = builtins.min
builtins_max = builtins.max
builtins_abs = builtins.abs
builtins_all = builtins.all
builtins_any = builtins.any


# === Array Creation Functions ===

def array(data, dtype=None) -> GFOArray:
    """Create a GFOArray from data."""
    return GFOArray(data, dtype=dtype)

def asarray(data, dtype=None) -> GFOArray:
    """Convert to GFOArray if not already."""
    if isinstance(data, GFOArray):
        return data if dtype is None else data.astype(dtype)
    return GFOArray(data, dtype=dtype)

def zeros(shape: Shape, dtype=float) -> GFOArray:
    """Create array of zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 1:
        return GFOArray([dtype(0)] * shape[0])
    elif len(shape) == 2:
        return GFOArray([[dtype(0)] * shape[1] for _ in range(shape[0])])
    raise ValueError(f"Shape {shape} not supported")


def zeros_like(a, dtype=None) -> GFOArray:
    """Create array of zeros with same shape as input."""
    if isinstance(a, GFOArray):
        shape = a.shape
    elif hasattr(a, 'shape'):
        shape = a.shape
    else:
        # Assume 1D list/tuple
        shape = (len(a),)
    if dtype is None:
        dtype = float
    return zeros(shape, dtype)


def ones(shape: Shape, dtype=float) -> GFOArray:
    """Create array of ones."""
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 1:
        return GFOArray([dtype(1)] * shape[0])
    elif len(shape) == 2:
        return GFOArray([[dtype(1)] * shape[1] for _ in range(shape[0])])
    raise ValueError(f"Shape {shape} not supported")

def empty(shape: Shape, dtype=float) -> GFOArray:
    """Create uninitialized array (filled with zeros for simplicity)."""
    return zeros(shape, dtype)

def full(shape: Shape, fill_value, dtype=None) -> GFOArray:
    """Create array filled with value."""
    if isinstance(shape, int):
        shape = (shape,)
    val = dtype(fill_value) if dtype else fill_value
    if len(shape) == 1:
        return GFOArray([val] * shape[0])
    elif len(shape) == 2:
        return GFOArray([[val] * shape[1] for _ in range(shape[0])])
    raise ValueError(f"Shape {shape} not supported")

def arange(start, stop=None, step=1, dtype=None) -> GFOArray:
    """Create array with evenly spaced values."""
    if stop is None:
        start, stop = 0, start
    result = []
    val = start
    while val < stop:
        result.append(val)
        val += step
    if dtype:
        result = [dtype(x) for x in result]
    return GFOArray(result)

def linspace(start, stop, num=50, endpoint=True, dtype=None) -> GFOArray:
    """Create array with evenly spaced values over interval."""
    if num < 0:
        raise ValueError("Number of samples must be non-negative")
    if num == 0:
        return GFOArray([])
    if num == 1:
        return GFOArray([float(start)])

    if endpoint:
        step = (stop - start) / (num - 1)
    else:
        step = (stop - start) / num

    result = [start + i * step for i in range(num)]
    if dtype:
        result = [dtype(x) for x in result]
    return GFOArray(result)

def eye(n: int, m: int = None, dtype=float) -> GFOArray:
    """Create identity matrix."""
    if m is None:
        m = n
    result = [[dtype(1) if i == j else dtype(0) for j in range(m)] for i in range(n)]
    return GFOArray(result)

def meshgrid(*arrays, indexing="xy"):
    """Create coordinate matrices from coordinate vectors."""
    if len(arrays) != 2:
        raise NotImplementedError("meshgrid only supports 2D")

    x = list(arrays[0]) if isinstance(arrays[0], GFOArray) else list(arrays[0])
    y = list(arrays[1]) if isinstance(arrays[1], GFOArray) else list(arrays[1])

    if indexing == "xy":
        xx = [x for _ in y]
        yy = [[yi] * len(x) for yi in y]
    else:  # "ij"
        xx = [[xi] * len(y) for xi in x]
        yy = [y for _ in x]

    return GFOArray(xx), GFOArray(yy)


# === Array Properties ===

def ndim(x) -> int:
    if isinstance(x, GFOArray):
        return x.ndim
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            return 2
        return 1
    return 0

def shape(x) -> Tuple[int, ...]:
    if isinstance(x, GFOArray):
        return x.shape
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            return (len(x), len(x[0]))
        return (len(x),)
    return ()


# === Element-wise Math Functions ===

def exp(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.exp(v) for v in x._get_flat()])
    return math.exp(x)

def log(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.log(v) for v in x._get_flat()])
    return math.log(x)

def log10(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.log10(v) for v in x._get_flat()])
    return math.log10(x)

def sqrt(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.sqrt(v) for v in x._get_flat()])
    return math.sqrt(x)

def abs(x):
    if isinstance(x, GFOArray):
        return GFOArray([builtins_abs(v) for v in x._get_flat()])
    return builtins_abs(x)

def power(x, p):
    if isinstance(x, GFOArray):
        return GFOArray([v ** p for v in x._get_flat()])
    return x ** p

def square(x):
    return power(x, 2)

def sin(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.sin(v) for v in x._get_flat()])
    return math.sin(x)

def cos(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.cos(v) for v in x._get_flat()])
    return math.cos(x)


# === Rounding and Clipping ===

def clip(x, a_min, a_max):
    """Clip values to range [a_min, a_max]."""
    if isinstance(x, GFOArray):
        # Handle array bounds
        if isinstance(a_max, GFOArray):
            a_max_list = a_max._get_flat()
            return GFOArray([builtins_max(a_min, builtins_min(v, m))
                           for v, m in zip(x._get_flat(), a_max_list)])
        return GFOArray([builtins_max(a_min, builtins_min(v, a_max)) for v in x._get_flat()])
    return builtins_max(a_min, builtins_min(x, a_max))

def rint(x):
    """Round to nearest integer."""
    if isinstance(x, GFOArray):
        return GFOArray([round(v) for v in x._get_flat()])
    return round(x)

def round(x, decimals=0):
    """Round to given decimals."""
    if isinstance(x, GFOArray):
        return GFOArray([builtins.round(v, decimals) for v in x._get_flat()])
    return builtins.round(x, decimals)

def floor(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.floor(v) for v in x._get_flat()])
    return math.floor(x)

def ceil(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.ceil(v) for v in x._get_flat()])
    return math.ceil(x)


# === Comparison and Logic ===

def maximum(x, y):
    """Element-wise maximum."""
    if isinstance(x, GFOArray):
        if isinstance(y, GFOArray):
            return GFOArray([builtins_max(a, b) for a, b in zip(x._get_flat(), y._get_flat())])
        return GFOArray([builtins_max(v, y) for v in x._get_flat()])
    if isinstance(y, GFOArray):
        return GFOArray([builtins_max(x, v) for v in y._get_flat()])
    return builtins_max(x, y)

def minimum(x, y):
    """Element-wise minimum."""
    if isinstance(x, GFOArray):
        if isinstance(y, GFOArray):
            return GFOArray([builtins_min(a, b) for a, b in zip(x._get_flat(), y._get_flat())])
        return GFOArray([builtins_min(v, y) for v in x._get_flat()])
    if isinstance(y, GFOArray):
        return GFOArray([builtins_min(x, v) for v in y._get_flat()])
    return builtins_min(x, y)

def greater(x, y): return GFOArray(x) > y
def less(x, y): return GFOArray(x) < y
def equal(x, y): return GFOArray(x) == y

def isnan(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.isnan(v) for v in x._get_flat()])
    return math.isnan(x)

def isinf(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.isinf(v) for v in x._get_flat()])
    return math.isinf(x)

def isfinite(x):
    if isinstance(x, GFOArray):
        return GFOArray([math.isfinite(v) for v in x._get_flat()])
    return math.isfinite(x)


# === Aggregation Functions ===

def sum(x, axis=None):
    if isinstance(x, GFOArray):
        return x.sum(axis=axis)
    return builtins_sum(x)

def mean(x, axis=None):
    if isinstance(x, GFOArray):
        return x.mean(axis=axis)
    return builtins_sum(x) / len(x)

def std(x, axis=None, ddof=0):
    if isinstance(x, GFOArray):
        return x.std(axis=axis, ddof=ddof)
    m = mean(x)
    return math.sqrt(builtins_sum((v - m) ** 2 for v in x) / (len(x) - ddof))

def var(x, axis=None, ddof=0):
    return std(x, axis=axis, ddof=ddof) ** 2

def prod(x, axis=None):
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)
    result = 1
    for v in flat:
        result *= v
    return result

def cumsum(x, axis=None):
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)
    result = []
    total = 0
    for v in flat:
        total += v
        result.append(total)
    return GFOArray(result)


# === Index Operations ===

def argmax(x, axis=None):
    if isinstance(x, GFOArray):
        return x.argmax(axis=axis)
    return list(x).index(builtins_max(x))

def argmin(x, axis=None):
    if isinstance(x, GFOArray):
        return x.argmin(axis=axis)
    return list(x).index(builtins_min(x))

def argsort(x, axis=-1):
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)
    return GFOArray(sorted(range(len(flat)), key=lambda i: flat[i]))

def where(condition, x=None, y=None):
    """Where with optional x/y values."""
    if isinstance(condition, GFOArray):
        cond_flat = condition._get_flat()
    else:
        cond_flat = list(condition)

    if x is None and y is None:
        # Return indices where True
        indices = [i for i, c in enumerate(cond_flat) if c]
        return (GFOArray(indices),)

    # Conditional selection
    if isinstance(x, GFOArray):
        x_flat = x._get_flat()
    elif isinstance(x, (list, tuple)):
        x_flat = list(x)
    else:
        x_flat = [x] * len(cond_flat)

    if isinstance(y, GFOArray):
        y_flat = y._get_flat()
    elif isinstance(y, (list, tuple)):
        y_flat = list(y)
    else:
        y_flat = [y] * len(cond_flat)

    result = [xv if c else yv for c, xv, yv in zip(cond_flat, x_flat, y_flat)]
    return GFOArray(result)

def nonzero(x):
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)
    indices = [i for i, v in enumerate(flat) if v]
    return (GFOArray(indices),)

def searchsorted(a, v, side="left"):
    """Find indices where elements should be inserted."""
    if isinstance(a, GFOArray):
        a_list = a._get_flat()
    else:
        a_list = list(a)

    if isinstance(v, (GFOArray, list, tuple)):
        v_list = list(v) if not isinstance(v, GFOArray) else v._get_flat()
        return GFOArray([_binary_search(a_list, val, side) for val in v_list])
    return _binary_search(a_list, v, side)

def _binary_search(arr, val, side):
    """Binary search for sorted insertion point."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if (side == "left" and arr[mid] < val) or (side == "right" and arr[mid] <= val):
            lo = mid + 1
        else:
            hi = mid
    return lo

def take(a, indices, axis=None):
    if isinstance(a, GFOArray):
        flat = a._get_flat()
    else:
        flat = list(a)

    if isinstance(indices, GFOArray):
        indices = indices._get_flat()
    elif isinstance(indices, (list, tuple)):
        indices = list(indices)
    else:
        return flat[indices]

    return GFOArray([flat[i] for i in indices])


# === Set Operations ===

def unique(x, return_index=False, return_inverse=False, return_counts=False):
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)

    seen = {}
    result = []
    for i, v in enumerate(flat):
        if v not in seen:
            seen[v] = len(result)
            result.append(v)

    result.sort()

    if not any([return_index, return_inverse, return_counts]):
        return GFOArray(result)

    # Build return tuple
    ret = [GFOArray(result)]
    if return_index:
        indices = [flat.index(v) for v in result]
        ret.append(GFOArray(indices))
    if return_inverse:
        result_list = result
        inverse = [result_list.index(v) for v in flat]
        ret.append(GFOArray(inverse))
    if return_counts:
        counts = [flat.count(v) for v in result]
        ret.append(GFOArray(counts))

    return tuple(ret)

def intersect1d(ar1, ar2):
    if isinstance(ar1, GFOArray):
        set1 = set(ar1._get_flat())
    else:
        set1 = set(ar1)

    if isinstance(ar2, GFOArray):
        set2 = set(ar2._get_flat())
    else:
        set2 = set(ar2)

    return GFOArray(sorted(set1 & set2))

def isin(element, test_elements):
    if isinstance(test_elements, GFOArray):
        test_set = set(test_elements._get_flat())
    else:
        test_set = set(test_elements)

    if isinstance(element, GFOArray):
        return GFOArray([v in test_set for v in element._get_flat()])
    return element in test_set


# === Array Manipulation ===

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
    result = []
    for arr in arrays:
        if isinstance(arr, GFOArray):
            if arr.ndim == 1:
                result.extend(arr._get_flat())
            else:
                result.extend(arr._data)
        else:
            result.extend(arr)
    return GFOArray(result)

def stack(arrays, axis=0):
    arrays_list = [list(a) if not isinstance(a, GFOArray) else a._get_flat() for a in arrays]
    return GFOArray(arrays_list)

def vstack(arrays):
    return stack(arrays, axis=0)

def hstack(arrays):
    result = []
    for arr in arrays:
        if isinstance(arr, GFOArray):
            result.extend(arr._get_flat())
        else:
            result.extend(arr)
    return GFOArray(result)

def tile(x, reps):
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)

    if isinstance(reps, int):
        return GFOArray(flat * reps)
    raise NotImplementedError("Multi-dimensional tile not supported")

def repeat(x, repeats, axis=None):
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)

    if isinstance(repeats, int):
        result = []
        for v in flat:
            result.extend([v] * repeats)
        return GFOArray(result)
    raise NotImplementedError("Variable repeats not supported")


def array_split(x, indices_or_sections, axis=0):
    """Split array into sub-arrays."""
    if isinstance(x, GFOArray):
        flat = x._get_flat()
    else:
        flat = list(x)

    n = len(flat)
    if isinstance(indices_or_sections, int):
        # Split into n equal parts
        n_sections = indices_or_sections
        section_size = n // n_sections
        remainder = n % n_sections

        result = []
        start = 0
        for i in range(n_sections):
            # Distribute remainder across first sections
            end = start + section_size + (1 if i < remainder else 0)
            result.append(GFOArray(flat[start:end]))
            start = end
        return result
    else:
        # Split at specified indices
        result = []
        prev = 0
        for idx in indices_or_sections:
            result.append(GFOArray(flat[prev:idx]))
            prev = idx
        result.append(GFOArray(flat[prev:]))
        return result


def split(x, indices_or_sections, axis=0):
    """Split array into sub-arrays (equal-sized splits only)."""
    return array_split(x, indices_or_sections, axis)


# === Linear Algebra ===

def dot(a, b):
    """Dot product of two arrays."""
    if isinstance(a, GFOArray):
        a_flat = a._get_flat()
    else:
        a_flat = list(a)

    if isinstance(b, GFOArray):
        b_flat = b._get_flat()
    else:
        b_flat = list(b)

    # 1D dot product
    if len(a_flat) == len(b_flat):
        return builtins_sum(x * y for x, y in zip(a_flat, b_flat))

    raise NotImplementedError("Matrix dot product not fully implemented")

def matmul(a, b):
    """Matrix multiplication (simplified for 2D)."""
    if isinstance(a, GFOArray) and isinstance(b, GFOArray):
        if a.ndim == 2 and b.ndim == 2:
            m, k1 = a.shape
            k2, n = b.shape
            if k1 != k2:
                raise ValueError("Incompatible dimensions")
            result = [[builtins_sum(a[i, k] * b[k, j] for k in range(k1))
                      for j in range(n)] for i in range(m)]
            return GFOArray(result)
    return dot(a, b)

def outer(a, b):
    """Outer product of two vectors."""
    if isinstance(a, GFOArray):
        a_flat = a._get_flat()
    else:
        a_flat = list(a)

    if isinstance(b, GFOArray):
        b_flat = b._get_flat()
    else:
        b_flat = list(b)

    return GFOArray([[x * y for y in b_flat] for x in a_flat])


class linalg:
    """Linear algebra namespace (simplified implementations)."""

    @staticmethod
    def solve(a, b):
        """Solve linear system Ax = b (simplified for small systems)."""
        # This is a very basic implementation - use for small systems only
        raise NotImplementedError(
            "linalg.solve requires scipy. Install scipy for full functionality."
        )

    @staticmethod
    def lstsq(a, b, rcond=None):
        raise NotImplementedError(
            "linalg.lstsq requires scipy. Install scipy for full functionality."
        )

    @staticmethod
    def pinv(a):
        raise NotImplementedError(
            "linalg.pinv requires scipy. Install scipy for full functionality."
        )

    @staticmethod
    def inv(a):
        raise NotImplementedError(
            "linalg.inv requires scipy. Install scipy for full functionality."
        )

    @staticmethod
    def eigvalsh(a):
        raise NotImplementedError(
            "linalg.eigvalsh requires scipy. Install scipy for full functionality."
        )

    @staticmethod
    def norm(x, ord=None, axis=None):
        """Vector/matrix norm."""
        if isinstance(x, GFOArray):
            flat = x._get_flat()
        else:
            flat = list(x)

        if ord is None or ord == 2:
            return math.sqrt(builtins_sum(v ** 2 for v in flat))
        elif ord == 1:
            return builtins_sum(builtins_abs(v) for v in flat)
        elif ord == inf:
            return builtins_max(builtins_abs(v) for v in flat)
        raise NotImplementedError(f"Norm order {ord} not implemented")

    @staticmethod
    def det(a):
        raise NotImplementedError(
            "linalg.det requires scipy. Install scipy for full functionality."
        )


# === Random Number Generation ===

class random:
    """Random number generation namespace."""

    _rng = py_random.Random()

    @staticmethod
    def seed(seed=None):
        random._rng = py_random.Random(seed)

    @staticmethod
    def randint(low, high=None, size=None):
        if high is None:
            low, high = 0, low

        if size is None:
            return random._rng.randint(low, high - 1)

        if isinstance(size, int):
            return GFOArray([random._rng.randint(low, high - 1) for _ in range(size)])

        # Multi-dimensional
        total = 1
        for s in size:
            total *= s
        flat = [random._rng.randint(low, high - 1) for _ in range(total)]
        return GFOArray(flat).reshape(size)

    @staticmethod
    def choice(a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a = list(range(a))
        elif isinstance(a, GFOArray):
            a = a._get_flat()
        else:
            a = list(a)

        if size is None:
            if p is not None:
                return random._rng.choices(a, weights=p, k=1)[0]
            return random._rng.choice(a)

        if isinstance(size, int):
            n = size
        else:
            n = 1
            for s in size:
                n *= s

        if replace:
            if p is not None:
                result = random._rng.choices(a, weights=p, k=n)
            else:
                result = [random._rng.choice(a) for _ in range(n)]
        else:
            result = random._rng.sample(a, n)

        return GFOArray(result)

    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        if size is None:
            return random._rng.uniform(low, high)

        if isinstance(size, int):
            return GFOArray([random._rng.uniform(low, high) for _ in range(size)])

        total = 1
        for s in size:
            total *= s
        flat = [random._rng.uniform(low, high) for _ in range(total)]
        return GFOArray(flat).reshape(size)

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        if size is None:
            return random._rng.gauss(loc, scale)

        if isinstance(size, int):
            return GFOArray([random._rng.gauss(loc, scale) for _ in range(size)])

        total = 1
        for s in size:
            total *= s
        flat = [random._rng.gauss(loc, scale) for _ in range(total)]
        return GFOArray(flat).reshape(size)

    @staticmethod
    def laplace(loc=0.0, scale=1.0, size=None):
        """Laplace distribution via inverse CDF."""
        def _laplace():
            u = random._rng.random() - 0.5
            return loc - scale * (1 if u >= 0 else -1) * math.log(1 - 2 * builtins_abs(u))

        if size is None:
            return _laplace()

        if isinstance(size, int):
            return GFOArray([_laplace() for _ in range(size)])

        total = 1
        for s in size:
            total *= s
        flat = [_laplace() for _ in range(total)]
        return GFOArray(flat).reshape(size)

    @staticmethod
    def logistic(loc=0.0, scale=1.0, size=None):
        """Logistic distribution via inverse CDF."""
        def _logistic():
            u = random._rng.random()
            return loc + scale * math.log(u / (1 - u))

        if size is None:
            return _logistic()

        if isinstance(size, int):
            return GFOArray([_logistic() for _ in range(size)])

        total = 1
        for s in size:
            total *= s
        flat = [_logistic() for _ in range(total)]
        return GFOArray(flat).reshape(size)

    @staticmethod
    def gumbel(loc=0.0, scale=1.0, size=None):
        """Gumbel distribution via inverse CDF."""
        def _gumbel():
            u = random._rng.random()
            return loc - scale * math.log(-math.log(u))

        if size is None:
            return _gumbel()

        if isinstance(size, int):
            return GFOArray([_gumbel() for _ in range(size)])

        total = 1
        for s in size:
            total *= s
        flat = [_gumbel() for _ in range(total)]
        return GFOArray(flat).reshape(size)

    @staticmethod
    def shuffle(x):
        if isinstance(x, GFOArray):
            random._rng.shuffle(x._data)
        else:
            random._rng.shuffle(x)

    @staticmethod
    def permutation(x):
        if isinstance(x, int):
            result = list(range(x))
        elif isinstance(x, GFOArray):
            result = x._get_flat().copy()
        else:
            result = list(x)
        random._rng.shuffle(result)
        return GFOArray(result)

    class RandomState:
        """RandomState for compatibility."""
        def __init__(self, seed=None):
            self._rng = py_random.Random(seed)

        def randint(self, low, high=None, size=None):
            if high is None:
                low, high = 0, low
            if size is None:
                return self._rng.randint(low, high - 1)
            return GFOArray([self._rng.randint(low, high - 1) for _ in range(size)])

        def choice(self, a, size=None, replace=True):
            if isinstance(a, int):
                a = list(range(a))
            if size is None:
                return self._rng.choice(a)
            if replace:
                return GFOArray([self._rng.choice(a) for _ in range(size)])
            return GFOArray(self._rng.sample(a, size))

        def uniform(self, low=0.0, high=1.0, size=None):
            if size is None:
                return self._rng.uniform(low, high)
            return GFOArray([self._rng.uniform(low, high) for _ in range(size)])

        def normal(self, loc=0.0, scale=1.0, size=None):
            if size is None:
                return self._rng.gauss(loc, scale)
            return GFOArray([self._rng.gauss(loc, scale) for _ in range(size)])


# === Utility Functions ===

def copy(x):
    if isinstance(x, GFOArray):
        return x.copy()
    return GFOArray(x)

def allclose(a, b, rtol=1e-5, atol=1e-8):
    if isinstance(a, GFOArray):
        a = a._get_flat()
    if isinstance(b, GFOArray):
        b = b._get_flat()

    for x, y in zip(a, b):
        if builtins_abs(x - y) > atol + rtol * builtins_abs(y):
            return False
    return True

def all(x, axis=None):
    if isinstance(x, GFOArray):
        return builtins_all(x._get_flat())
    return builtins_all(x)

def any(x, axis=None):
    if isinstance(x, GFOArray):
        return builtins_any(x._get_flat())
    return builtins_any(x)

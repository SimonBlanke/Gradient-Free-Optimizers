"""
NumPy backend - thin wrapper around NumPy for unified interface.

This module re-exports NumPy functions used by GFO, providing a consistent
interface that can be swapped with the pure Python backend.
"""

import numpy as np

# === Array Creation ===
array = np.array
asarray = np.asarray
zeros = np.zeros
zeros_like = np.zeros_like
ones = np.ones
empty = np.empty
empty_like = np.empty_like
full = np.full
arange = np.arange
linspace = np.linspace
meshgrid = np.meshgrid
eye = np.eye
diag = np.diag

# === Array Properties ===
ndim = np.ndim
shape = lambda x: np.shape(x)

# === Type Conversion ===
int32 = np.int32
int64 = np.int64
float32 = np.float32
float64 = np.float64

# === Mathematical Operations ===
sum = np.sum
mean = np.mean
std = np.std
var = np.var
prod = np.prod
cumsum = np.cumsum

# === Element-wise Math ===
exp = np.exp
log = np.log
log10 = np.log10
sqrt = np.sqrt
abs = np.abs
power = np.power
square = np.square
sin = np.sin
cos = np.cos

# === Rounding and Clipping ===
clip = np.clip
rint = np.rint
round = np.round
floor = np.floor
ceil = np.ceil

# === Comparison and Logic ===
maximum = np.maximum
minimum = np.minimum
greater = np.greater
less = np.less
equal = np.equal
isnan = np.isnan
isinf = np.isinf
isfinite = np.isfinite

# === Index Operations ===
argmax = np.argmax
argmin = np.argmin
argsort = np.argsort
where = np.where
nonzero = np.nonzero
searchsorted = np.searchsorted
take = np.take

# === Set Operations ===
unique = np.unique
intersect1d = np.intersect1d
isin = np.isin

# === Array Manipulation ===
reshape = np.reshape
transpose = np.transpose
ravel = np.ravel
flatten = lambda x: np.asarray(x).flatten()
concatenate = np.concatenate
stack = np.stack
vstack = np.vstack
hstack = np.hstack
tile = np.tile
repeat = np.repeat
array_split = np.array_split
split = np.split

# === Linear Algebra ===
dot = np.dot
matmul = np.matmul
outer = np.outer


class linalg:
    """Linear algebra namespace."""

    solve = staticmethod(np.linalg.solve)
    lstsq = staticmethod(np.linalg.lstsq)
    pinv = staticmethod(np.linalg.pinv)
    inv = staticmethod(np.linalg.inv)
    eigvalsh = staticmethod(np.linalg.eigvalsh)
    norm = staticmethod(np.linalg.norm)
    det = staticmethod(np.linalg.det)


# === Random Number Generation ===
class random:
    """Random number generation namespace."""

    seed = staticmethod(np.random.seed)
    randint = staticmethod(np.random.randint)
    choice = staticmethod(np.random.choice)
    uniform = staticmethod(np.random.uniform)
    normal = staticmethod(np.random.normal)
    laplace = staticmethod(np.random.laplace)
    logistic = staticmethod(np.random.logistic)
    gumbel = staticmethod(np.random.gumbel)
    shuffle = staticmethod(np.random.shuffle)
    permutation = staticmethod(np.random.permutation)
    RandomState = np.random.RandomState


# === Constants ===
inf = np.inf
pi = np.pi
e = np.e
nan = np.nan

# === Utility ===
copy = np.copy
allclose = np.allclose
all = np.all
any = np.any

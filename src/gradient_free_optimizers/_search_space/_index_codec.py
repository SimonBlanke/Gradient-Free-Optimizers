"""Index-based codec translating Space points <-> pure integer arrays.

The **IndexCodec** is a lightweight façade that turns every value coming
from a `Space` instance into the integer *index* it occupies inside the
corresponding `Dimension` – and back again.  The goal is to let any
optimizer treat the search-space as a dense numeric hyper-rectangle while
keeping round-trip cost *O(D)* and memory footprint minimal.

New in *v2* (July 2025)
=======================
* **`to_dict()`** – convenience helper that converts any *tuple/array*
  of raw values into a ``{name: value}`` mapping so that callers do *not*
  have to remember the global ordering of dimensions.

Example
-------
>>> spec = {"units": [16, 32, 64],    # categorical
...         "lr": (-5.0, -1.0),        # real (log-uniform grid)
...         "use_bn": [True, False]}   # bool = categorical
>>> from _search_space._space import Space
>>> space = Space(spec)
>>> codec = IndexCodec(space)
>>> p = space.sample(); p                       # array(dtype=object)
array([32, -2.931, True], dtype=object)
>>> codec.to_dict(p)
{'units': 32, 'lr': -2.931, 'use_bn': True}
>>> idx = codec.encode(p); idx                  # int32 indexes
array([1, 157, 0], dtype=int32)
>>> codec.decode(idx, as_dict=True)
{'units': 32, 'lr': 0.00117, 'use_bn': True}

The integer bounds for every axis are available via :pyattr:`max_index`.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Iterable, Any
import math
import numpy as np

from ._space import Space

__all__ = ["IndexCodec"]


class IndexCodec:
    """Cheap‑and‑cheerful integer *codec* for a :class:`~_search_space._space.Space`.

    Parameters
    ----------
    space : Space
        A fully‑constructed search‑space instance (see ``_search_space/_space.py``).
    res   : int, default=256
        Resolution used for *continuous* dimensions when falling back to
        the legacy ``value_to_pos`` helper (categorical / integer
        dimensions ignore this argument).
    """

    # --------------------------------------------------------------------- construction
    def __init__(self, search_space, *, res: int = 256):
        self.space = Space(search_space)
        self.res = int(res)

        # Cache vectorised helpers for speed ---------------------------------------------
        self._pos_to_val = tuple(getattr(d, "pos_to_value") for d in self.space.dims)
        self._val_to_pos = tuple(getattr(d, "value_to_pos") for d in self.space.dims)

        # Fast access to per‑dimension size / max index ----------------------------------
        self._sizes = np.fromiter(
            (
                (
                    int(getattr(d, "size", math.inf))
                    if math.isfinite(getattr(d, "size", math.inf))
                    else self.res
                )  # uncountable → treat as *virtual* grid of length *res*
                for d in self.space.dims
            ),
            dtype=np.int32,
            count=len(self.space.dims),
        )
        self._max_index = self._sizes - 1

    # --------------------------------------------------------------------- dict helper
    def to_dict(self, point: Mapping[str, Any] | Sequence[Any]):
        """Return *point* as an **ordered dict** keyed by dimension names.

        This is a thin convenience layer: `Space.perturb()` or user code
        often deals with tuples / NumPy arrays.  Many callers, however,
        prefer **named** access when building model configs.
        """
        if isinstance(point, Mapping):
            # Already a mapping – normalise order and return copy
            return {name: point[name] for name in self.space.names}

        if len(point) != len(self.space.dims):
            raise ValueError(
                "Point length mismatch (expected %d)." % len(self.space.dims)
            )

        return {name: val for name, val in zip(self.space.names, point)}

    # --------------------------------------------------------------------- core API
    def encode(self, point: Mapping[str, Any] | Sequence[Any]) -> np.ndarray:
        """Translate *one* point into its integer‑index representation."""
        ordered = (
            [point[n] for n in self.space.names]
            if isinstance(point, Mapping)
            else point
        )
        if len(ordered) != len(self.space.dims):
            raise ValueError(
                "Point length mismatch (expected %d)." % len(self.space.dims)
            )

        return np.fromiter(
            (
                f(v, self.res) if f.__code__.co_argcount == 3 else f(v)
                for f, v in zip(self._val_to_pos, ordered)
            ),
            dtype=np.int32,
            count=len(self.space.dims),
        )

    def decode(self, idx: Sequence[int], *, as_dict: bool = False):
        """Translate *one* integer vector back to original values."""
        if len(idx) != len(self.space.dims):
            raise ValueError(
                "Index length mismatch (expected %d)." % len(self.space.dims)
            )

        values = [
            f(int(i), self.res) if f.__code__.co_argcount == 3 else f(int(i))
            for f, i in zip(self._pos_to_val, idx)
        ]
        return self.space._vectorise(values, as_dict=as_dict)

    # ............................................ batch variants ------------------------
    def encode_many(self, points: Iterable[Mapping[str, Any] | Sequence[Any]]):
        return np.vstack([self.encode(p) for p in points])

    def decode_many(self, indices: np.ndarray, *, as_dict: bool = False):
        return [self.decode(idx, as_dict=as_dict) for idx in indices]

    # ...................................................................................
    @property
    def sizes(self) -> np.ndarray:
        """Per‑dimension *discrete* sizes (unbounded → *res*)."""
        return self._sizes.copy()

    @property
    def max_index(self) -> np.ndarray:
        """Vector of inclusive upper bounds for every axis."""
        return self._max_index.copy()

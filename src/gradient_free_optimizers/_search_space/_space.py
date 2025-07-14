# search_space.py – revised Space wrapper (dict-only constructor)
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Mapping, List, Dict, Sequence
import math
import random
import numpy as np

from ._dimension import Dimension, make_dimension

__all__ = ["Space"]


class Space:
    """Vectorised façade around a *search‑space dictionary*.

    Example
    -------
    >>> raw = {"x": (-10, 10), "act": ["relu", "tanh"], "lr": st.loguniform(1e-5, 1e-2)}
    >>> space = Space(raw)
    >>> point = space.sample()             # ndarray shape (d,)
    >>> neighbourhood = space.perturb(point, scale=0.05)
    """

    # ---------------------------------------------------------------------
    def __init__(self, search_space: Mapping[str, Any]):
        if not search_space:
            raise ValueError("Search‑space dictionary must not be empty.")

        self.names: List[str] = list(search_space.keys())
        self._dims: List[Dimension] = [
            make_dimension(search_space[k]) for k in self.names
        ]
        self.dims: tuple[Dimension, ...] = tuple(self._dims)  # read‑only tuple

    # ------------------------------------------------------------------ repr
    def __repr__(self):  # pragma: no cover
        pairs = ", ".join(f"{n}={d}" for n, d in zip(self.names, self.dims))
        return f"Space({pairs})"

    # ---------------------------------------------------------------- convenience magic
    def __len__(self):
        return len(self.dims)

    def __iter__(self):  # iterate over Dimension objects (order = self.names)
        return iter(self.dims)

    # ---------------------------------------------------------------- internal helper
    def _vectorise(self, items, *, as_dict: bool):
        return (
            dict(zip(self.names, items)) if as_dict else np.asarray(items, dtype=object)
        )

    # ---------------------------------------------------------------- public API
    # ............................... sample
    def sample(
        self,
        n: int | None = None,
        *,
        rng: random.Random | np.random.Generator = random,
        as_dict: bool = False,
    ):
        if n is None or n == 1:
            vals = [d.sample(rng) for d in self.dims]
            return self._vectorise(vals, as_dict=as_dict)
        out = np.empty((n, len(self)), dtype=object)
        for i in range(n):
            for j, d in enumerate(self.dims):
                out[i, j] = d.sample(rng)
        if as_dict:
            return [dict(zip(self.names, row)) for row in out]
        return out

    # ............................... grid
    def grid(self, per_dim: int = 16) -> Dict[str, np.ndarray]:
        """Return *per-dimension* grids (no cartesian product)."""
        return {n: d.grid(per_dim) for n, d in zip(self.names, self.dims)}

    # ............................... span & size (vector properties)
    @property
    def span(self) -> np.ndarray:
        return np.asarray([d.span for d in self.dims], dtype=float)

    @property
    def size(self) -> np.ndarray:
        return np.asarray(
            [getattr(d, "size", math.inf) for d in self.dims], dtype=float
        )

    # ............................... clip
    def clip(self, point, *, as_dict: bool = False):
        if isinstance(point, Mapping):
            ordered = [point[k] for k in self.names]
        else:
            if len(point) != len(self):
                raise ValueError("Point length mismatch → expected %d." % len(self))
            ordered = point
        clipped = [dim.clip(val) for dim, val in zip(self.dims, ordered)]
        return self._vectorise(clipped, as_dict=as_dict)

    # ............................... perturb
    def perturb(
        self,
        point,
        *,
        scale: float = 0.1,
        rng: random.Random | np.random.Generator = np.random,
        as_dict: bool = False,
    ):
        if isinstance(point, Mapping):
            ordered = [point[k] for k in self.names]
        else:
            if len(point) != len(self):
                raise ValueError("Point length mismatch → expected %d." % len(self))
            ordered = point
        perturbed = [d.perturb(v, scale, rng) for d, v in zip(self.dims, ordered)]
        return self._vectorise(perturbed, as_dict=as_dict)

    # ............................... distance
    def distance(self, p, q, *, assume_clipped: bool = False) -> float:
        if isinstance(p, Mapping):
            p_vals = [p[n] for n in self.names]
            q_vals = [q[n] for n in self.names]
        else:
            if len(p) != len(self) or len(q) != len(self):
                raise ValueError("Point length mismatch.")
            p_vals, q_vals = p, q
        total = 0.0
        for dim, a, b in zip(self.dims, p_vals, q_vals):
            if not assume_clipped:
                a, b = dim.clip(a), dim.clip(b)
            if dim.span == 0:
                continue
            if hasattr(dim, "low") and hasattr(dim, "high"):
                delta = (float(a) - float(b)) / dim.span
            else:
                delta = 0.0 if a == b else 1.0
            total += delta * delta
        return math.sqrt(total)

    # ---------------------------------------------------------------- legacy shims
    @property  # pragma: no cover
    def max_positions(self):
        raise AttributeError("'max_positions' is obsolete.  Use 'space.size'.")

    @property  # pragma: no cover
    def dim_sizes(self):
        raise AttributeError("'dim_sizes' removed.  Use 'space.size'.")

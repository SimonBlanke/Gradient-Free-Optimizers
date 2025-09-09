from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gradient_free_optimizers._result import Result
from .dimensions import BaseDimension, CategoricalDimension
from .converter_v2 import converter_from_search_space


@dataclass
class V2CompatConverter:
    """Compatibility wrapper exposing the legacy Converter API but using z-space.

    - position == z-vector (floats in [0,1])
    - value    == native values per dimension
    - para     == dict mapping names->values
    """

    names: Sequence[str]
    dims: Sequence[BaseDimension]
    constraints: Optional[List] = None

    def __post_init__(self) -> None:
        self._conv = converter_from_search_space(self.names, self.dims)
        self.para_names = list(self.names)
        self.n_dimensions = len(self.dims)
        self.constraints = self.constraints or []

        # Legacy attributes (not meaningful for v2, but kept for compatibility)
        self.max_positions = np.ones(self.n_dimensions, dtype=float)
        self.search_space_positions = [
            [0.0, 1.0] for _ in range(self.n_dimensions)
        ]
        self.search_space_size = np.inf

    # Constraints API
    def not_in_constraint(self, position: Sequence[float]) -> bool:
        params = self.value2para(self.position2value(position))
        for constraint in self.constraints:
            if not constraint(params):
                return False
        return True

    # Conversions
    def position2value(self, position: Optional[Sequence[float]]):
        if position is None:
            return None
        return self._conv.z_to_values(position)

    def value2position(self, value: Optional[Sequence[Any]]):
        if value is None:
            return None
        return np.array(self._conv.values_to_z(value), dtype=float)

    def value2para(self, value: Optional[Sequence[Any]]):
        if value is None:
            return None
        return self._conv.values_to_params(value)

    def para2value(self, para: Optional[Dict[str, Any]]):
        if para is None:
            return None
        return self._conv.params_to_values(para)

    def values2positions(self, values: Optional[List[Sequence[Any]]] = None):
        if values is None:
            return None
        return [self.value2position(v) for v in values]

    def positions2values(self, positions: Optional[List[Sequence[float]]] = None):
        if positions is None:
            return None
        return [self.position2value(p) for p in positions]

    def values2paras(self, values: List[Sequence[Any]]):
        return [self.value2para(v) for v in values]

    # Memory/DataFrame helpers (keys are tuple(z) for compatibility)
    def positions_scores2memory_dict(self, positions: List[Sequence[float]], scores: List[float]):
        pos_keys = [tuple(map(float, p)) for p in positions]
        results = [Result(float(s), {}) for s in scores]
        return dict(zip(pos_keys, results))

    def memory_dict2positions_scores(self, memory_dict: Dict[Tuple[float, ...], Result]):
        positions = [np.array(list(k), dtype=float) for k in memory_dict.keys()]
        scores = [res.score if isinstance(res, Result) else float(res) for res in memory_dict.values()]
        return positions, scores

    def dataframe2memory_dict(self, dataframe: Optional[pd.DataFrame]):
        if dataframe is None or dataframe.empty:
            return {}
        parameter = set(self.para_names)
        memory_para = set(dataframe.columns)
        if not (parameter <= memory_para and "score" in dataframe.columns):
            return {}
        values = list(dataframe[self.para_names].values)
        positions = self.values2positions(values)
        scores = dataframe["score"].tolist()
        return self.positions_scores2memory_dict(positions, scores)

    def memory_dict2dataframe(self, memory_dict: Dict[Tuple[float, ...], Result]):
        positions, scores = self.memory_dict2positions_scores(memory_dict)
        values = self.positions2values(positions)
        df = pd.DataFrame(values, columns=self.para_names)
        df["score"] = scores
        return df


class V2Initializer:
    """Initializer that generates z-vectors based on initialize config.

    Supports keys: "random", "grid", "vertices", "warm_start".
    """

    def __init__(self, conv: V2CompatConverter, initialize: Dict[str, Any]):
        self.conv = conv
        self.initialize = initialize
        self.init_positions_l: List[np.ndarray] = []
        self.n_inits = 0
        self._build()

    def _build(self):
        inits: List[np.ndarray] = []
        if self.initialize.get("random", 0):
            inits.extend(self._random(self.initialize["random"]))
        if self.initialize.get("grid", 0):
            inits.extend(self._grid(self.initialize["grid"]))
        if self.initialize.get("vertices", 0):
            inits.extend(self._vertices(self.initialize["vertices"]))
        if self.initialize.get("warm_start"):
            inits.extend(self._warm_start(self.initialize["warm_start"]))

        self.init_positions_l = inits
        self.n_inits = len(self.init_positions_l)

        # Backfill with random if counts mismatch later
        target = self._get_target_n_inits(self.initialize)
        if self.n_inits < target:
            self.init_positions_l.extend(self._random(target - self.n_inits))
            self.n_inits = len(self.init_positions_l)

    @staticmethod
    def _get_target_n_inits(initialize: Dict[str, Any]) -> int:
        n = 0
        for k, v in initialize.items():
            if k == "warm_start" and hasattr(v, "__len__"):
                n += len(v)
            elif isinstance(v, int):
                n += v
        return n

    def _random(self, n: int) -> List[np.ndarray]:
        rng = np.random.default_rng()
        out: List[np.ndarray] = []
        for _ in range(n):
            z = self.conv._conv.sample_z(rng)
            out.append(np.array(z, dtype=float))
        return out

    def _grid(self, n_points: int) -> List[np.ndarray]:
        # Build per-dim z grids and combine until cap ~ n_points
        per_dim = max(2, int(round(n_points ** (1.0 / max(1, self.conv.n_dimensions)))))
        grids = self.conv._conv.grid_z(per_dim)
        mesh = np.array(np.meshgrid(*grids, indexing="ij"))
        points = mesh.T.reshape(-1, self.conv.n_dimensions)
        # limit to n_points
        return [points[i] for i in range(min(n_points, len(points)))]

    def _vertices(self, n: int) -> List[np.ndarray]:
        # Randomly sample corners in {0,1}^d without many duplicates
        rng = np.random.default_rng()
        out: List[np.ndarray] = []
        seen = set()
        tries = 0
        while len(out) < n and tries < n * 100:
            corner = rng.integers(0, 2, size=self.conv.n_dimensions).astype(float)
            key = tuple(corner.tolist())
            if key not in seen:
                seen.add(key)
                out.append(corner)
            tries += 1
        # Fill remaining with random if needed
        if len(out) < n:
            out.extend(self._random(n - len(out)))
        return out

    def _warm_start(self, value_list: List[Dict[str, Any]]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for para in value_list:
            values = self.conv.para2value(para)
            z = self.conv.value2position(values)
            out.append(np.array(z, dtype=float))
        return out

    # Compatibility with legacy Initializer API
    def add_n_random_init_pos(self, n: int) -> None:
        if n <= 0:
            return
        self.init_positions_l.extend(self._random(n))
        self.n_inits = len(self.init_positions_l)

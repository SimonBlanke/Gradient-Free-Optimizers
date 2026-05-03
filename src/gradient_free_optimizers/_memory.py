# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from __future__ import annotations

import logging
from multiprocessing.managers import DictProxy
from typing import TYPE_CHECKING, Any

from ._objective_adapter import ObjectiveAdapter

if TYPE_CHECKING:
    import pandas as pd
from ._storage import BaseStorage, MemoryStorage


class CachedObjectiveAdapter(ObjectiveAdapter):
    def __init__(
        self, conv, objective, storage: BaseStorage | None = None, optimizer_ref=None
    ):
        super().__init__(conv, objective, optimizer_ref=optimizer_ref)

        self._storage: BaseStorage = storage if storage is not None else MemoryStorage()

    @property
    def memory_dict(self):
        """Backward-compatible access to the underlying storage.

        Returns the raw dict for MemoryStorage backends. For other backends,
        returns the storage object itself (supports ``in`` and ``[]``
        operations via the BaseStorage interface, but is not a true dict).
        """
        if isinstance(self._storage, MemoryStorage):
            return self._storage._data
        return self._storage

    def memory(self, warm_start: pd.DataFrame, memory: Any = None):
        # DictProxy from multiprocessing.Manager: wrap it in MemoryStorage
        if isinstance(memory, DictProxy):
            self._storage = MemoryStorage(data=memory)

        if warm_start is None:
            return

        import pandas as pd

        if not isinstance(warm_start, pd.DataFrame):
            logging.warning("Memory warm start must be of type pandas.DataFrame")
            logging.warning("Optimization will continue without memory warm start")
            return

        if warm_start.empty:
            logging.warning("Memory warm start has no values in current search space")
            logging.warning("Optimization will continue without memory warm start")
            return

        warm_dict = self._conv.dataframe2memory_dict(warm_start)
        if warm_dict:
            self._storage.update(warm_dict)

    def __call__(self, pos):
        pos_t = tuple(pos)

        if self._storage.contains(pos_t):
            full_params = self._conv.value2para(self._conv.position2value(pos))
            if self._conv.conditions:
                active_mask = self._conv.evaluate_conditions(full_params)
            else:
                active_mask = None
            return self._storage.get(pos_t), full_params, active_mask
        else:
            result, full_params, active_mask = self._call_objective(pos)
            self._storage.put(pos_t, result)
            return result, full_params, active_mask

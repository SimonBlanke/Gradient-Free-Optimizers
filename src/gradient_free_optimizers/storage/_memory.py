"""In-memory storage backend using a plain dictionary.

This is the default storage backend, equivalent to the original behavior
of ``memory=True``. It also accepts dict-like objects (e.g. multiprocessing
DictProxy) for cross-process sharing on a single machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ._base import BaseStorage

if TYPE_CHECKING:
    from .._result import Result


class MemoryStorage(BaseStorage):
    """In-memory evaluation cache backed by a Python dict.

    Parameters
    ----------
    data : dict-like or None, default=None
        An existing dict (or multiprocessing.managers.DictProxy) to use
        as the backing store. If None, a fresh dict is created.
        This allows sharing evaluation caches between processes when
        a DictProxy from a multiprocessing.Manager is provided.

    Examples
    --------
    Default usage (created automatically by ``memory=True``)::

        storage = MemoryStorage()
        opt.search(objective, n_iter=100, memory=storage)

    Sharing across processes via DictProxy::

        from multiprocessing import Manager
        manager = Manager()
        shared_dict = manager.dict()
        storage = MemoryStorage(data=shared_dict)
    """

    def __init__(self, data: dict[tuple, Any] | None = None):
        self._data: dict = data if data is not None else {}

    def get(self, key: tuple) -> Result | None:
        return self._data.get(key)

    def put(self, key: tuple, result: Result) -> None:
        self._data[key] = result

    def contains(self, key: tuple) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Iterator[tuple[tuple, Result]]:
        yield from self._data.items()

    def update(self, mapping: dict) -> None:
        self._data.update(mapping)

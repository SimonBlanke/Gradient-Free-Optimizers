"""Pluggable storage backends for evaluation caching.

Storage backends control where objective function evaluation results are
persisted. They can be passed to the ``memory`` parameter of ``search()``::

    from gradient_free_optimizers._storage import SQLiteStorage

    storage = SQLiteStorage("results.db")
    opt.search(objective, n_iter=100, memory=storage)

The default behavior (``memory=True``) uses :class:`MemoryStorage`, an
in-memory dict that matches the original caching behavior.
"""

from ._base import BaseStorage
from ._memory import MemoryStorage
from ._sqlite import SQLiteStorage

__all__ = ["BaseStorage", "MemoryStorage", "SQLiteStorage"]

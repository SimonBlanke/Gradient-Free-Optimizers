"""Abstract base class for evaluation storage backends.

Storage backends provide a key-value interface for caching objective function
evaluations. Positions (as tuples of indices) map to Result objects containing
the score and any associated metrics.

All stored scores use optimizer-internal scale (negated for minimization).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._result import Result


class BaseStorage(ABC):
    """Abstract base for evaluation result storage.

    Subclass this and implement all abstract methods to create a custom
    storage backend. Storage backends can be passed to the ``memory``
    parameter of :meth:`~gradient_free_optimizers.search.Search.search`
    to control where evaluation results are cached.

    Keys are position tuples (integer indices into the search space).
    Values are :class:`~gradient_free_optimizers._result.Result` objects.

    Parameters
    ----------
    None

    Examples
    --------
    Creating a custom backend::

        class RedisStorage(BaseStorage):
            def __init__(self, url):
                self._client = redis.Redis.from_url(url)
            def get(self, key):
                ...
    """

    @abstractmethod
    def get(self, key: tuple) -> Result | None:
        """Return the cached Result for a position, or None if not cached.

        Parameters
        ----------
        key : tuple
            Position indices as a tuple of integers.

        Returns
        -------
        Result or None
        """
        ...

    @abstractmethod
    def put(self, key: tuple, result: Result) -> None:
        """Store a Result for a position.

        Parameters
        ----------
        key : tuple
            Position indices as a tuple of integers.
        result : Result
            The evaluation result to store.
        """
        ...

    @abstractmethod
    def contains(self, key: tuple) -> bool:
        """Check whether a position has been evaluated.

        Parameters
        ----------
        key : tuple
            Position indices as a tuple of integers.

        Returns
        -------
        bool
        """
        ...

    def update(self, mapping: dict) -> None:
        """Bulk-insert from a dictionary of {key: Result} pairs.

        The default implementation calls :meth:`put` in a loop. Subclasses
        may override for more efficient bulk operations.

        Parameters
        ----------
        mapping : dict
            Dictionary mapping position tuples to Result objects.
        """
        for key, result in mapping.items():
            self.put(key, result)

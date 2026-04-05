"""Base class for distributed evaluation backends.

Provides the Template Method Pattern for distribution strategies.
Subclasses implement _distribute() to define how objective function
evaluations are spread across workers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    """Abstract base for distributing objective function evaluations.

    Subclass this and implement _distribute() to create a custom backend.
    Use the .distribute decorator on an objective function to enable
    parallel batch evaluation during optimization.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers for evaluation.

    Examples
    --------
    Creating a custom backend::

        class MyBackend(BaseDistribution):
            def _distribute(self, func, params_batch):
                return [my_remote_eval(func, p) for p in params_batch]

        @MyBackend(n_workers=4).distribute
        def objective(x, y):
            return -(x**2 + y**2)
    """

    def __init__(self, n_workers: int):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers

    @abstractmethod
    def _distribute(self, func, params_batch: list[dict]) -> list[float]:
        """Evaluate func(**params) for each params dict in the batch.

        This is the only method subclasses need to implement. It receives
        the original (unwrapped) objective function and a list of parameter
        dictionaries, and must return a list of scores in the same order.

        The function follows GFO's convention: ``func(params_dict)`` where
        params_dict is a single dictionary, not keyword arguments.

        If an evaluation fails, raise the exception. Error handling
        is done by the search loop via its ``catch`` parameter.

        Parameters
        ----------
        func : callable
            The original objective function with signature f(dict) -> float.
        params_batch : list[dict]
            Parameter dictionaries to evaluate.

        Returns
        -------
        list[float]
            Scores in the same order as params_batch.
        """
        ...

    def distribute(self, func):
        """Decorator that wraps a single-point objective for batch evaluation.

        The decorated function accepts a list of parameter dicts and returns
        a list of scores. It also carries metadata attributes that the
        optimizer's search loop uses to detect and configure batch mode.

        Parameters
        ----------
        func : callable
            Objective function with signature f(**params) -> float.

        Returns
        -------
        callable
            Wrapped function with signature f(list[dict]) -> list[float].
        """

        def wrapper(params_batch):
            return self._distribute(func, params_batch)

        wrapper.__name__ = getattr(func, "__name__", "objective")

        # Metadata for search.py to detect batch mode
        wrapper._gfo_distributed = True
        wrapper._gfo_batch_size = self.n_workers
        wrapper._gfo_original_func = func

        return wrapper

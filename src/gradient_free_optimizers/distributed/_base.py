"""Base class for distributed evaluation backends.

Provides the Template Method Pattern for distribution strategies.
Subclasses implement _distribute() to define how objective function
evaluations are spread across workers. Async-capable backends
additionally implement _submit() and _wait_any().
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    """Abstract base for distributing objective function evaluations.

    Subclass this and implement _distribute() to create a custom backend.
    Use the .distribute decorator on an objective function to enable
    parallel batch evaluation during optimization.

    For async-capable backends, also implement _submit() and _wait_any()
    and set ``_is_async = True``. Async backends enable true asynchronous
    evaluation where results are processed as they arrive, keeping all
    workers busy at all times.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers for evaluation.

    Examples
    --------
    Creating a custom sync backend::

        class MyBackend(BaseDistribution):
            def _distribute(self, func, params_batch):
                return [my_remote_eval(func, p) for p in params_batch]

        @MyBackend(n_workers=4).distribute
        def objective(x, y):
            return -(x**2 + y**2)

    Creating a custom async backend::

        class MyAsyncBackend(BaseDistribution):
            _is_async = True

            def _distribute(self, func, params_batch):
                futures = [self._submit(func, p) for p in params_batch]
                return [self._get_result(f) for f in futures]

            def _submit(self, func, params):
                return my_remote_submit(func, params)

            def _wait_any(self, futures):
                done = my_wait_for_any(futures)
                return done, self._get_result(done)
    """

    _is_async = False

    def __init__(self, n_workers: int):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers

    @abstractmethod
    def _distribute(self, func, params_batch: list[dict]) -> list:
        """Evaluate func(params) for each params dict in the batch.

        This is the only method subclasses need to implement for
        synchronous (batch) evaluation. It receives the original
        (unwrapped) objective function and a list of parameter
        dictionaries, and must return a list of results in the same order.

        The function follows GFO's convention: ``func(params_dict)`` where
        params_dict is a single dictionary, not keyword arguments.

        Parameters
        ----------
        func : callable
            The original objective function with signature
            f(dict) -> float or f(dict) -> (float, dict).
        params_batch : list[dict]
            Parameter dictionaries to evaluate.

        Returns
        -------
        list[float | tuple[float, dict]]
            Results in the same order as params_batch. Each element is
            either a plain score or a (score, metrics) tuple, matching
            whatever the objective function returns.
        """
        ...

    def _submit(self, func, params: dict):
        """Submit a single evaluation asynchronously.

        Only required for async backends (``_is_async = True``).
        Returns a backend-specific future object that can be passed
        to :meth:`_wait_any`.

        Parameters
        ----------
        func : callable
            The original objective function.
        params : dict
            Single parameter dictionary to evaluate.

        Returns
        -------
        future
            A backend-specific future/handle object.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support async evaluation. "
            f"Set _is_async = True and implement _submit() and _wait_any()."
        )

    def _wait_any(self, futures) -> tuple:
        """Wait for any submitted future to complete.

        Only required for async backends (``_is_async = True``).
        Blocks until at least one future from the collection is ready,
        then returns both the future object and its raw result.

        Parameters
        ----------
        futures : iterable
            Collection of future objects from :meth:`_submit`.

        Returns
        -------
        tuple of (future, float | tuple[float, dict])
            The completed future object and the objective function's
            return value (a plain score or a (score, metrics) tuple).
            The future is returned so the caller can identify which
            submission completed (e.g., to look up the associated position).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support async evaluation. "
            f"Set _is_async = True and implement _submit() and _wait_any()."
        )

    def distribute(self, func):
        """Decorator that wraps a single-point objective for batch evaluation.

        The decorated function accepts a list of parameter dicts and returns
        a list of results. It also carries metadata attributes that the
        optimizer's search loop uses to detect and configure batch mode.

        Parameters
        ----------
        func : callable
            Objective function with signature
            f(params_dict) -> float or f(params_dict) -> (float, dict).

        Returns
        -------
        callable
            Wrapped function with signature
            f(list[dict]) -> list[float | tuple[float, dict]].
        """

        def wrapper(params_batch):
            return self._distribute(func, params_batch)

        wrapper.__name__ = getattr(func, "__name__", "objective")

        # Metadata for search.py to detect and configure distribution
        wrapper._gfo_distributed = True
        wrapper._gfo_batch_size = self.n_workers
        wrapper._gfo_original_func = func
        wrapper._gfo_backend = self

        return wrapper

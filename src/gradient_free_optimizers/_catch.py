from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class _CatchWrapper:
    """Callable that catches exceptions from the objective and returns fallbacks.

    A class instead of a closure so it can be pickled by multiprocessing's
    spawn context (Windows, macOS 3.14+).
    """

    def __init__(self, func: Callable, catch: dict[type[Exception], int | float]):
        self.func = func
        self.catch = catch
        self._catch_types = tuple(catch.keys())

    def __call__(self, params):
        try:
            return self.func(params)
        except self._catch_types as e:
            for exc_type, fallback_score in self.catch.items():
                if isinstance(e, exc_type):
                    logger.warning(
                        "Caught %s in objective function: %s. "
                        "Using fallback score: %s",
                        type(e).__name__,
                        e,
                        fallback_score,
                    )
                    return fallback_score
            raise


def wrap_with_catch(
    objective_function: Callable,
    catch: dict[type[Exception], int | float],
) -> Callable:
    """Wrap objective function to catch exceptions and return fallback scores."""
    return _CatchWrapper(objective_function, catch)

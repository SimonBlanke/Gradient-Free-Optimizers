from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def wrap_with_catch(
    objective_function: Callable,
    catch: dict[type[Exception], int | float],
) -> Callable:
    """Wrap objective function to catch exceptions and return fallback scores."""
    catch_types = tuple(catch.keys())

    def wrapped(params):
        try:
            return objective_function(params)
        except catch_types as e:
            for exc_type, fallback_score in catch.items():
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

    return wrapped

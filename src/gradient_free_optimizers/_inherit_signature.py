# source: https://stackoverflow.com/questions/77255184/inheriting-all-init-arguments-type-hints-from-parent-class

from typing import ParamSpec, TypeVar, Callable
from functools import update_wrapper

P = ParamSpec("P")
T = TypeVar("T")


def inherit_signature(
    original: Callable[P, T]
) -> Callable[[Callable], Callable[P, T]]:
    """Set the signature of one function to the signature of another."""

    def wrapper(f: Callable) -> Callable[P, T]:
        return update_wrapper(f, original)

    return wrapper

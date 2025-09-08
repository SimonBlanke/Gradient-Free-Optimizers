from dataclasses import dataclass
import sys
from pathlib import Path

# Ensure project src is on sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradient_free_optimizers._search_space.base import BaseSearchSpace, combine_spaces
from gradient_free_optimizers._search_space.converter_v2 import (
    converter_from_search_space,
)


@dataclass
class A(BaseSearchSpace):
    a: tuple = (0.0, 1.0)
    k: list = ("x", "y")


@dataclass
class B(BaseSearchSpace):
    b: tuple = (1, 5)


def test_combined_space_build():
    cs = combine_spaces(A(), B())
    names, dims, offsets = cs.build()
    assert names[0] == "__variant__"
    assert dims[0].kind == "categorical"
    # Ensure variants added
    assert set(names[1:]) == {"a", "k", "b"}

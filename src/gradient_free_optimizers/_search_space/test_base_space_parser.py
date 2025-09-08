from dataclasses import dataclass, field
import sys
from pathlib import Path

# Ensure project src is on sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy.stats as st

from gradient_free_optimizers._search_space.base import BaseSearchSpace
from gradient_free_optimizers._search_space.converter_v2 import (
    converter_from_search_space,
)


@dataclass
class MySpace(BaseSearchSpace):
    x: tuple = (-10.0, 10.0)
    lr: object = st.loguniform(1e-5, 1e-2)
    filters: object = field(default_factory=lambda: np.arange(16, 65, 16))
    act: list = field(default_factory=lambda: ["relu", "gelu", "tanh"])
    dropout: object = st.beta(2, 5)
    use_bn: list = field(default_factory=lambda: [True, False])
    seed: int = 42


def test_build_dimensions_and_sampling():
    space = MySpace()
    names, dims = space._build_dimensions()
    assert names[0] == "x"
    assert len(names) == len(dims) == 7

    conv = converter_from_search_space(names, dims)
    z = conv.sample_z()
    assert len(z) == len(dims)

    values = conv.z_to_values(z)
    params = conv.values_to_params(values)
    assert set(params.keys()) == set(names)
    assert isinstance(params["x"], float)
    assert isinstance(params["filters"], (int, float))
    assert params["seed"] == 42

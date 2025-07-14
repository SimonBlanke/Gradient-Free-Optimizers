# tests/test_space_and_dimensions.py
from __future__ import annotations

import math
import types
import random

import numpy as np
import pytest

from ._dimension import (
    make_dimension,
    FixedDimension,
    CategoricalDimension,
    RealDimension,
    IntegerDimension,
    DistributionDimension,
)
from ._space import Space

try:
    import scipy.stats as st
except ModuleNotFoundError:  # pragma: no cover
    st = None


# --------------------------------------------------------------------------- #
# Helper fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rng():
    return random.Random(1234)


@pytest.fixture
def example_space():
    spec = {
        "x": (-10.0, 10.0),  # Real
        "filters": range(16, 128 + 1, 16),  # Categorical
        "act": ["relu", "gelu", "tanh"],  # Categorical
        "dropout": (0.0, 1.0),  # Real
        "use_bn": [True, False],  # Categorical
        "seed": 42,  # Fixed
    }
    if st is not None:
        spec["lr"] = st.loguniform(1e-5, 1e-2)  # Distribution
    return Space(spec)


# --------------------------------------------------------------------------- #
# Dimension unit tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, cls",
    [
        (42, FixedDimension),
        ([0, 1, 2], CategoricalDimension),
        ((-1.0, 1.0), RealDimension),
        ((-3, 3), IntegerDimension),
        (range(4), CategoricalDimension),
    ],
)
def test_make_dimension_inference(raw, cls):
    dim = make_dimension(raw)
    assert isinstance(dim, cls)


def test_fixed_dimension():
    d = FixedDimension("foo")
    assert d.sample() == "foo"
    assert d.span == 0
    assert d.perturb("foo", 0.5) == "foo"
    assert d.clip("bar") == "foo"


def test_categorical_dimension():
    d = CategoricalDimension(["a", "b", "c"])
    assert d.span == 2
    assert d.clip("z") in {"a", "b", "c"}
    assert d.perturb("a", 1.0) in {"a", "b", "c"}


def test_real_dimension_log():
    d = RealDimension(1e-5, 1e-2, log=True)
    s = d.sample()
    assert 1e-5 <= s <= 1e-2
    assert math.isclose(d.span, 1e-2 - 1e-5, rel_tol=0, abs_tol=1e-5)
    assert d.clip(-1) == 1e-5
    g = d.grid(10)
    assert g[0] == pytest.approx(1e-5)
    assert g[-1] == pytest.approx(1e-2)


def test_integer_dimension():
    d = IntegerDimension(3, 7)
    for _ in range(20):
        assert 3 <= d.sample() <= 7
    assert d.grid().tolist() == list(range(3, 8))
    assert d.clip(100) == 7


"""
@pytest.mark.skipif(st is None, reason="scipy not available")
def test_distribution_dimension():
    dist = st.beta(2, 5)
    d = DistributionDimension(dist)
    s = d.sample()
    assert 0.0 <= s <= 1.0
    assert math.isfinite(d.span)
    assert 0.0 < d.clip(-10) < 1.0
"""

# --------------------------------------------------------------------------- #
# Space unit tests
# --------------------------------------------------------------------------- #


def test_space_basic_properties(example_space):
    d = len(example_space)
    assert d >= 6
    assert isinstance(example_space.span, np.ndarray)
    assert example_space.span.shape == (d,)


def test_space_sample_shapes(example_space, rng):
    one = example_space.sample(rng=rng)
    assert one.shape == (len(example_space),)

    five = example_space.sample(5, rng=rng)
    assert five.shape == (5, len(example_space))

    one_dict = example_space.sample(rng=rng, as_dict=True)
    assert set(one_dict.keys()) == set(example_space.names)


def test_space_clip_and_perturb(example_space):
    point = example_space.sample()
    clipped = example_space.clip(point)
    assert clipped.shape == point.shape

    # push point clearly outside bounds for numeric dims
    bad = point.copy()
    bad[0] = 1e9  # assumes dim0 is real
    clipped2 = example_space.clip(bad)
    assert clipped2[0] != 1e9

    neighbour = example_space.perturb(point, scale=0.2)
    print("\n neighbour", neighbour, "\n")
    assert neighbour.shape == point.shape
    assert example_space.distance(point, neighbour) >= 0.0

    assert False


def test_space_distance_symmetry(example_space):
    p = example_space.sample()
    q = example_space.sample()
    d1 = example_space.distance(p, q)
    d2 = example_space.distance(q, p)
    assert pytest.approx(d1) == d2
    assert example_space.distance(p, p) == 0.0


"""
def test_space_grid(example_space):
    grids = example_space.grid(per_dim=11)
    assert isinstance(grids, dict)
    assert set(grids.keys()) == set(example_space.names)
    # every grid is at most 11 points

    for value in grids.values():
        print("value", value)

    assert all(g.size <= 11 for g in grids.values())
"""

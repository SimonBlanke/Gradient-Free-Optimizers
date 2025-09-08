import math
import sys
from pathlib import Path

# Ensure project src is on sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy.stats as st

from gradient_free_optimizers._search_space.dimensions import (
    CategoricalDimension,
    DistributionDimension,
    IntegerDimension,
    RealDimension,
)


def test_real_dimension_mapping_and_perturb():
    d = RealDimension(-10.0, 10.0)
    assert math.isclose(d.value_to_z(-10.0), 0.0)
    assert math.isclose(d.value_to_z(10.0), 1.0)
    assert math.isclose(d.z_to_value(0.25), -5.0)

    v = 0.0
    v2 = d.perturb(v, scale=0.1, rng=np.random.default_rng(0))
    assert -10.0 <= v2 <= 10.0


def test_integer_dimension_rounding_and_grid():
    d = IntegerDimension(2, 6, brackets="[]")
    assert d.z_to_value(0.0) == 2
    assert d.z_to_value(1.0) == 6
    # Middle should be around 4
    assert d.z_to_value(0.5) in {4, 5}
    g = d.grid(3)
    # Unique, sorted grid
    assert g == sorted(set(g))
    assert min(g) >= 2 and max(g) <= 6


def test_categorical_dimension_index_and_perturb():
    d = CategoricalDimension(["relu", "gelu", "tanh"])
    assert d.z_to_value(0.0) == "relu"
    assert d.z_to_value(0.999999) == "tanh"
    assert d.value_to_z("gelu") > 0.33 and d.value_to_z("gelu") < 0.67
    v2 = d.perturb("gelu", scale=0.0, rng=np.random.default_rng(1))
    assert v2 in {"relu", "tanh"}


def test_distribution_dimension_quantile_mapping():
    rv = st.beta(2, 5)
    d = DistributionDimension(rv)
    # z=0.5 should map near the median
    med = rv.ppf(0.5)
    assert abs(d.z_to_value(0.5) - med) < 1e-6
    # Value roundtrip
    val = d.z_to_value(0.2)
    z = d.value_to_z(val)
    assert 0.0 <= z <= 1.0


def test_real_dimension_brackets_affect_grid_and_sample():
    d1 = RealDimension(-1.0, 1.0, brackets="[]")
    d2 = RealDimension(-1.0, 1.0, brackets="()")
    g1 = d1.grid(5)
    g2 = d2.grid(5)
    assert min(g1) == -1.0 and max(g1) == 1.0
    assert min(g2) > -1.0 and max(g2) < 1.0
    # sample does not hit excluded endpoints
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = d2.sample(rng)
        assert -1.0 < s < 1.0


def test_integer_dimension_brackets_exact():
    d = IntegerDimension(2, 6, brackets="()")
    assert d.size == 3  # 3,4,5
    vals = set(d.grid(10))
    assert vals == {3, 4, 5}
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = d.sample(rng)
        assert s in {3, 4, 5}

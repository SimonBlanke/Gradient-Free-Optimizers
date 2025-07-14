import math
import numpy as np
import scipy.stats.distributions as st
import pytest

from gradient_free_optimizers._search_space._base_search_space import BaseSearchSpace


# --------------------------------------------------------------------------- example
class ToySpace(BaseSearchSpace):
    x: np.ndarray = np.arange(-10, 10, 0.1)
    lr: st.rv_frozen = st.loguniform(1e-5, 1e-2)
    filters: np.ndarray = np.arange(16, 129, 16)
    act: list[str] = ["relu", "gelu", "tanh"]
    dropout: st.rv_frozen = st.beta(2, 5)
    use_bn: list[bool] = [True, False]
    seed: int = 42


# --------------------------------------------------------------------------- tests
def test_init_and_to_dict():
    ss = ToySpace()
    d = ss.to_dict()
    assert d["seed"] == 42
    assert len(d["filters"]) == 8


def test_assignment_good():
    ss = ToySpace()
    ss.use_bn = [False, True]


def test_assignment_bad():
    ss = ToySpace()
    with pytest.raises(TypeError):
        ss.lr = 0.1  # wrong type


def test_constructor_type_check():
    with pytest.raises(TypeError):
        ToySpace(seed="oops")


def test_mutability_isolated():
    a = ToySpace()
    b = ToySpace()
    a.act.append("selu")
    assert "selu" not in b.act  # each instance got its own copy


class ToySpace(BaseSearchSpace):
    x: np.ndarray = np.arange(-10, 10, 0.1)
    lr: st.rv_frozen = st.loguniform(1e-5, 1e-2)
    filters: np.ndarray = np.arange(16, 129, 16)
    act: list[str] = ["relu", "gelu", "tanh"]
    dropout: st.rv_frozen = st.beta(2, 5)
    use_bn: list[bool] = [True, False]
    seed: int = 42


class NumericSpace(BaseSearchSpace):
    a: np.ndarray = np.linspace(0.0, 1.0, 11)
    b: list[int] = [1, 2, 3]
    c: int = 5


class DistributionSpace(BaseSearchSpace):
    lr: st.rv_frozen = st.loguniform(1e-5, 1e-2)
    dropout: st.rv_frozen = st.beta(2, 5)


def test_toyspace_metadata():
    ss = ToySpace()
    assert ss.parameter_names == [
        "x",
        "lr",
        "filters",
        "act",
        "dropout",
        "use_bn",
        "seed",
    ]
    assert ss.n_dimensions == 7
    assert ss.dimension_types[0] is np.ndarray
    assert math.isinf(ss.dimension_sizes[1])
    assert ss.dimension_sizes[0] == 200
    assert ss.dimension_sizes[5] == 2
    assert ss.dimension_sizes[-1] == 1


def test_numericspace_metadata():
    ss = NumericSpace()
    assert ss.parameter_names == ["a", "b", "c"]
    assert ss.n_dimensions == 3
    assert ss.dimension_sizes == [11, 3, 1]


def test_distributionspace_metadata():
    ss = DistributionSpace()
    assert ss.parameter_names == ["lr", "dropout"]
    assert ss.n_dimensions == 2
    assert all(math.isinf(size) for size in ss.dimension_sizes)

import time
import pytest
import numpy as np
import pandas as pd
from gradient_free_optimizers.converter import Converter


def equal_arraysInList(list1, list2):
    return all((e1 == e2).all() for e1, e2 in zip(list1, list2))


######### test position2value #########


position2value_test_para_0 = [
    (np.array([0]), np.array([-10])),
    (np.array([20]), np.array([10])),
    (np.array([10]), np.array([0])),
]


@pytest.mark.parametrize("test_input,expected", position2value_test_para_0)
def test_position2value_0(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    conv = Converter(search_space)
    value = conv.position2value(test_input)

    assert value == expected


position2value_test_para_1 = [
    (np.array([0, 0]), np.array([-10, 0])),
    (np.array([20, 0]), np.array([10, 0])),
    (np.array([10, 10]), np.array([0, 10])),
]


@pytest.mark.parametrize("test_input,expected", position2value_test_para_1)
def test_position2value_1(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
        "x2": np.arange(0, 11, 1),
    }

    conv = Converter(search_space)
    value = conv.position2value(test_input)

    assert (value == expected).all()


######### test value2position #########


value2position_test_para_0 = [
    (np.array([-10]), np.array([0])),
    (np.array([10]), np.array([20])),
    (np.array([0]), np.array([10])),
]


@pytest.mark.parametrize("test_input,expected", value2position_test_para_0)
def test_value2position_0(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    conv = Converter(search_space)
    position = conv.value2position(test_input)

    assert position == expected


value2position_test_para_1 = [
    (np.array([-10, 11]), np.array([0, 10])),
    (np.array([10, 11]), np.array([20, 10])),
    (np.array([0, 0]), np.array([10, 0])),
]


@pytest.mark.parametrize("test_input,expected", value2position_test_para_1)
def test_value2position_1(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
        "x2": np.arange(0, 11, 1),
    }

    conv = Converter(search_space)
    position = conv.value2position(test_input)

    assert (position == expected).all()


######### test value2para #########


value2para_test_para_0 = [
    (np.array([-10]), {"x1": np.array([-10])}),
    (np.array([10]), {"x1": np.array([10])}),
    (np.array([0]), {"x1": np.array([0])}),
]


@pytest.mark.parametrize("test_input,expected", value2para_test_para_0)
def test_value2para_0(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    conv = Converter(search_space)
    para = conv.value2para(test_input)

    assert para == expected


value2para_test_para_1 = [
    (np.array([-10, 11]), {"x1": np.array([-10]), "x2": np.array([11])}),
    (np.array([10, 11]), {"x1": np.array([10]), "x2": np.array([11])}),
    (np.array([0, 0]), {"x1": np.array([0]), "x2": np.array([0])}),
]


@pytest.mark.parametrize("test_input,expected", value2para_test_para_1)
def test_value2para_1(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
        "x2": np.arange(0, 11, 1),
    }

    conv = Converter(search_space)
    para = conv.value2para(test_input)

    assert para == expected


######### test para2value #########


para2value_test_para_0 = [
    ({"x1": np.array([-10])}, np.array([-10])),
    ({"x1": np.array([10])}, np.array([10])),
    ({"x1": np.array([0])}, np.array([0])),
]


@pytest.mark.parametrize("test_input,expected", para2value_test_para_0)
def test_para2value_0(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    conv = Converter(search_space)
    value = conv.para2value(test_input)

    assert value == expected


para2value_test_para_1 = [
    ({"x1": np.array([-10]), "x2": np.array([11])}, np.array([-10, 11])),
    ({"x1": np.array([10]), "x2": np.array([11])}, np.array([10, 11])),
    ({"x1": np.array([0]), "x2": np.array([0])}, np.array([0, 0])),
]


@pytest.mark.parametrize("test_input,expected", para2value_test_para_1)
def test_para2value_1(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
        "x2": np.arange(0, 11, 1),
    }

    conv = Converter(search_space)
    value = conv.para2value(test_input)

    assert (value == expected).all()


######### test values2positions #########


values_0 = [
    np.array([-10]),
    np.array([10]),
    np.array([0]),
]

positions_0 = [
    np.array([0]),
    np.array([20]),
    np.array([10]),
]


values_1 = [
    np.array([-10]),
    np.array([10]),
    np.array([0]),
    np.array([-10]),
    np.array([10]),
    np.array([0]),
    np.array([-10]),
    np.array([10]),
    np.array([0]),
]

positions_1 = [
    np.array([0]),
    np.array([20]),
    np.array([10]),
    np.array([0]),
    np.array([20]),
    np.array([10]),
    np.array([0]),
    np.array([20]),
    np.array([10]),
]


values2positions_test_para_0 = [
    (values_0, positions_0),
    (values_1, positions_1),
]


@pytest.mark.parametrize("test_input,expected", values2positions_test_para_0)
def test_values2positions_0(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    conv = Converter(search_space)
    positions = conv.values2positions(test_input)

    assert positions == expected


values_0 = [
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
]

positions_0 = [
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
]


values_1 = [
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
]

positions_1 = [
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
]


values2positions_test_para_1 = [
    (values_0, positions_0),
    (values_1, positions_1),
]


@pytest.mark.parametrize("test_input,expected", values2positions_test_para_1)
def test_values2positions_1(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
        "x2": np.arange(0, 11, 1),
    }

    conv = Converter(search_space)
    positions = conv.values2positions(test_input)

    assert equal_arraysInList(positions, expected)


######### test positions2values #########


values_0 = [
    np.array([-10]),
    np.array([10]),
    np.array([0]),
]

positions_0 = [
    np.array([0]),
    np.array([20]),
    np.array([10]),
]


values_1 = [
    np.array([-10]),
    np.array([10]),
    np.array([0]),
    np.array([-10]),
    np.array([10]),
    np.array([0]),
    np.array([-10]),
    np.array([10]),
    np.array([0]),
]

positions_1 = [
    np.array([0]),
    np.array([20]),
    np.array([10]),
    np.array([0]),
    np.array([20]),
    np.array([10]),
    np.array([0]),
    np.array([20]),
    np.array([10]),
]


positions2values_test_para_0 = [
    (positions_0, values_0),
    (positions_1, values_1),
]

"""
@pytest.mark.parametrize("test_input,expected", positions2values_test_para_0)
def test_positions2values_0(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
    }

    conv = Converter(search_space)
    values = conv.positions2values(test_input)

    assert values == expected


values_0 = [
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
]

positions_0 = [
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
]


values_1 = [
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
    np.array([-10, 11]),
    np.array([10, 11]),
    np.array([0, 0]),
]

positions_1 = [
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
    np.array([0, 10]),
    np.array([20, 10]),
    np.array([10, 0]),
]


positions2values_test_para_1 = [
    (positions_0, values_0),
    # (positions_1, values_1),
]


@pytest.mark.parametrize("test_input,expected", positions2values_test_para_1)
def test_positions2values_1(test_input, expected):
    search_space = {
        "x1": np.arange(-10, 11, 1),
        "x2": np.arange(0, 11, 1),
    }

    conv = Converter(search_space)
    values = conv.positions2values(test_input)

    print("test_input", test_input)
    print("values", values)
    print("expected", expected)
    print("equal_arraysInList(values, expected)", equal_arraysInList(values, expected))

    assert equal_arraysInList(values, expected)
"""


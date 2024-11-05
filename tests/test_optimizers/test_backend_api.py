import pytest
import random
import numpy as np

from gradient_free_optimizers.optimizers.core_optimizer.converter import Converter

from ._parametrize import optimizers


def objective_function(para):
    return -(para["x1"] + para["x1"])


search_space1 = {
    "x1": np.array([1]),
    "x2": np.arange(-10, 10, 1),
}

search_space2 = {
    "x1": np.arange(-10, 10, 1),
    "x2": np.array([1]),
}

search_space3 = {
    "x1": np.arange(-10, 10, 1),
    "x2": np.array([1]),
    "x3": np.array([1]),
}

search_space4 = {
    "x1": np.arange(-10, 10, 1),
    "x2": np.array([1]),
    "x3": np.array([1]),
    "x4": np.array([1]),
}


objective_para = (
    "search_space",
    [
        (search_space1),
        (search_space2),
        (search_space3),
        (search_space4),
    ],
)


@pytest.mark.parametrize(*objective_para)
@pytest.mark.parametrize(*optimizers)
def test_backend_api_0(Optimizer, search_space):
    opt = Optimizer(search_space)

    conv = Converter(search_space)

    n_inits = len(opt.init.init_positions_l)

    for _ in range(n_inits):
        pos = opt.init_pos()
        value = conv.position2value(pos)
        para = conv.value2para(value)
        score = objective_function(para)
        opt.evaluate(score)

    opt.finish_initialization()

    for _ in range(20):
        pos = opt.iterate()
        value = conv.position2value(pos)
        para = conv.value2para(value)
        score = objective_function(para)
        opt.evaluate(score)

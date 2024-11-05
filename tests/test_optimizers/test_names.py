import time
import pytest
import numpy as np

from ._parametrize import optimizers


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 100, 1),
}


@pytest.mark.parametrize(*optimizers)
def test_name_0(Optimizer):
    opt = Optimizer(search_space)

    opt.name
    opt._name_
    opt.__name__

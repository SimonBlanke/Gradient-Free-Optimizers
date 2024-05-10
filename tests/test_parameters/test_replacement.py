# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np


from gradient_free_optimizers import (
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    LipschitzOptimizer,
    ForestOptimizer,
)


def parabola_function(para):
    loss = para["x"] * para["x"] + para["y"] * para["y"]
    return -loss


search_space = {
    "x": np.arange(-1, 1, 1),
    "y": np.arange(-1, 1, 1),
}

optimizer_para = (
    "optimizer",
    [
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (LipschitzOptimizer),
        (ForestOptimizer),
    ],
)


@pytest.mark.parametrize(*optimizer_para)
def test_replacement_0(optimizer):
    opt = optimizer(search_space, replacement=True)
    opt.search(parabola_function, n_iter=15)

    opt_false = optimizer(search_space, replacement=False)
    with pytest.raises((ValueError, IndexError)):
        opt_false.search(parabola_function, n_iter=15)
    assert len(opt_false.all_pos_comb) == 0

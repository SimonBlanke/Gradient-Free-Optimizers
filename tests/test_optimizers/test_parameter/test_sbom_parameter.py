# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
import numpy as np

from gradient_free_optimizers import (
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    DecisionTreeOptimizer,
    EnsembleOptimizer,
)


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-10, 11, 1)}


sbom_para = [
    ({"warm_start_smbo": None}),
    ({"warm_start_smbo": None}),
    ({"rand_rest_p": 0}),
    ({"rand_rest_p": 0.5}),
    ({"rand_rest_p": 1}),
    ({"rand_rest_p": 10}),
]


pytest_wrapper = ("para", sbom_para)

optimizers_sbom = (
    "Optimizer",
    [
        (BayesianOptimizer),
        (TreeStructuredParzenEstimators),
        (DecisionTreeOptimizer),
        (EnsembleOptimizer),
    ],
)


@pytest.mark.parametrize(*optimizers_sbom)
@pytest.mark.parametrize(*pytest_wrapper)
def test_smbo_para(Optimizer, para):
    opt = Optimizer(search_space, **para)
    opt.search(
        objective_function,
        n_iter=10,
        memory=False,
        verbosity=False,
        initialize={"vertices": 2},
    )

    for optimizer in opt.optimizers:
        para_key = list(para.keys())[0]
        para_value = getattr(optimizer, para_key)

        assert para_value == para[para_key]

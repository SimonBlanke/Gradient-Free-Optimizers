# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from gradient_free_optimizers import ParticleSwarmOptimizer
from ._base_test import _base_test

n_iter = 100
opt = ParticleSwarmOptimizer


def test_inertia():
    for inertia in [0.1, 0.9]:
        opt_para = {"inertia": inertia}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_cognitive_weight():
    for cognitive_weight in [0.1, 0.9]:
        opt_para = {"cognitive_weight": cognitive_weight}
        _base_test(opt, n_iter, opt_para=opt_para)


def test_social_weight():
    for social_weight in [0.1, 0.9]:
        opt_para = {"social_weight": social_weight}
        _base_test(opt, n_iter, opt_para=opt_para)

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .exp_imp_based_opt import ExpectedImprovementBasedOptimization

from .surrogate_models import (
    GPR_linear,
    GPR,
)

gaussian_process = {"gp_nonlinear": GPR(), "gp_linear": GPR_linear()}


class BayesianOptimizer(ExpectedImprovementBasedOptimization):
    def __init__(
        self,
        search_space,
        gpr=gaussian_process["gp_linear"],
        xi=0.01,
        warm_start_sbom=None,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space)
        self.regr = gpr
        self.xi = xi
        self.warm_start_sbom = warm_start_sbom
        self.rand_rest_p = rand_rest_p

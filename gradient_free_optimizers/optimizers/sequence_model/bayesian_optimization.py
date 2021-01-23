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
        initialize={"grid": 4, "random": 2, "vertices": 4},
        gpr=gaussian_process["gp_nonlinear"],
        xi=0.03,
        warm_start_smbo=None,
        rand_rest_p=0.03,
    ):
        super().__init__(search_space, initialize)
        self.gpr = gpr
        self.regr = gpr
        self.xi = xi
        self.warm_start_smbo = warm_start_smbo
        self.rand_rest_p = rand_rest_p

        self.init_warm_start_smbo()

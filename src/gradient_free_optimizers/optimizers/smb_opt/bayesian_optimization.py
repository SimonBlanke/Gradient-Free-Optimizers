# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from gradient_free_optimizers._array_backend import array as gfo_array, random as np_random

from .smbo import SMBO
from .surrogate_models import GPR
from .acquisition_function import ExpectedImprovement


def normalize(arr):
    arr = gfo_array(arr)
    arr_min = arr.min()
    arr_max = arr.max()
    range_ = arr_max - arr_min

    if range_ == 0:
        return np_random.uniform(0, 1, size=arr.shape)
    else:
        return (arr - arr_min) / range_


class BayesianOptimizer(SMBO):
    name = "Bayesian Optimization"
    _name_ = "bayesian_optimization"
    __name__ = "BayesianOptimizer"

    optimizer_type = "sequential"
    computationally_expensive = True

    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        constraints=[],
        random_state=None,
        rand_rest_p=0,
        nth_process=None,
        warm_start_smbo=None,
        max_sample_size=10000000,
        sampling={"random": 1000000},
        replacement=True,
        gpr=None,
        xi=0.03,
    ):
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            warm_start_smbo=warm_start_smbo,
            max_sample_size=max_sample_size,
            sampling=sampling,
            replacement=replacement,
        )

        # Instantiate GPR - supports both class and instance for backwards compatibility
        if gpr is None:
            self.gpr = GPR()
        elif isinstance(gpr, type):
            # User passed a class, instantiate it
            self.gpr = gpr()
        else:
            # User passed an instance
            self.gpr = gpr
        self.regr = self.gpr
        self.xi = xi

    def finish_initialization(self):
        self.all_pos_comb = self._all_possible_pos()
        return super().finish_initialization()

    def _expected_improvement(self):
        self.pos_comb = self._sampling(self.all_pos_comb)

        acqu_func = ExpectedImprovement(self.regr, self.pos_comb, self.xi)
        return acqu_func.calculate(self.X_sample, self.Y_sample)

    def _training(self):
        X_sample = gfo_array(self.X_sample)
        Y_sample = gfo_array(self.Y_sample)

        Y_sample = normalize(Y_sample).reshape(-1, 1)
        self.regr.fit(X_sample, Y_sample)

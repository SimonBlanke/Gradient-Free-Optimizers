# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from numpy.random import normal, laplace, logistic, gumbel

from .surrogate_models import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GPR_linear,
    GPR,
)


def merge_dicts(base_dict, added_dict):
    # overwrite default values
    for key in base_dict.keys():
        if key in list(added_dict.keys()):
            base_dict[key] = added_dict[key]

    return base_dict


gaussian_process = {"gp_nonlinear": GPR(), "gp_linear": GPR_linear()}

tree_regressor = {
    "random_forest": RandomForestRegressor(),
    "extra_tree": ExtraTreesRegressor(),
}


def skip_refit_75(i):
    if i <= 33:
        return 1
    return int((i - 33) ** 0.75)


def skip_refit_50(i):
    if i <= 33:
        return 1
    return int((i - 33) ** 0.5)


def skip_refit_25(i):
    if i <= 33:
        return 1
    return int((i - 33) ** 0.25)


def never_skip_refit(i):
    return 1


skip_retrain_ = {
    "many": skip_refit_75,
    "some": skip_refit_50,
    "few": skip_refit_25,
    "never": never_skip_refit,
}

distribution = {
    "normal": normal,
    "laplace": laplace,
    "logistic": logistic,
    "gumbel": gumbel,
}


class Arguments:
    def __init__(self, *args, **kwargs):
        kwargs_opt = {
            # HillClimbingOptimizer
            "epsilon": 0.05,
            "distribution": "normal",
            "n_neighbours": 1,
            "n_positions": 10,
            # StochasticHillClimbingOptimizer
            "p_down": 0.3,
            # TabuOptimizer
            "tabu_memory": 3,
            # RandomRestartHillClimbingOptimizer
            "n_iter_restart": 10,
            # RandomAnnealingOptimizer
            "annealing_rate": 0.99,
            # SimulatedAnnealingOptimizer
            "start_temp": 100,
            "norm_factor": "adaptive",
            # StochasticTunnelingOptimizer
            "gamma": 0.5,
            "warm_start_population": None,
            # ParallelTemperingOptimizer
            "system_temperatures": [0.1, 1, 10, 100],
            "n_iter_swap": 10,
            # ParticleSwarmOptimizer
            "n_particles": 10,
            "inertia": 0.5,
            "cognitive_weight": 0.5,
            "social_weight": 0.5,
            # EvolutionStrategyOptimizer
            "individuals": 10,
            "mutation_rate": 0.7,
            "crossover_rate": 0.3,
            # BayesianOptimizer
            "max_sample_size": 1000000,
            "warm_start_smbo": None,
            "xi": 0.01,
            "gpr": "gp_nonlinear",
            "skip_retrain": "never",
            # TreeStructuredParzenEstimators
            "start_up_evals": 10,
            "gamma_tpe": 0.3,
            "tree_regressor": "random_forest",
        }

        self.kwargs_opt = merge_dicts(kwargs_opt, kwargs)

    def set_opt_args(self):
        self.epsilon = self.kwargs_opt["epsilon"]
        self.distribution = distribution[self.kwargs_opt["distribution"]]
        self.n_neighbours = self.kwargs_opt["n_neighbours"]
        self.n_positions = self.kwargs_opt["n_positions"]

        self.p_down = self.kwargs_opt["p_down"]

        self.tabu_memory = self.kwargs_opt["tabu_memory"]

        self.n_iter_restart = self.kwargs_opt["n_iter_restart"]

        self.annealing_rate = self.kwargs_opt["annealing_rate"]
        self.start_temp = self.kwargs_opt["start_temp"]
        self.norm_factor = self.kwargs_opt["norm_factor"]
        self.gamma = self.kwargs_opt["gamma"]

        self.system_temperatures = self.kwargs_opt["system_temperatures"]
        self.n_iter_swap = self.kwargs_opt["n_iter_swap"]

        self.n_particles = self.kwargs_opt["n_particles"]
        self.inertia = self.kwargs_opt["inertia"]
        self.cognitive_weight = self.kwargs_opt["cognitive_weight"]
        self.social_weight = self.kwargs_opt["social_weight"]

        self.individuals = self.kwargs_opt["individuals"]
        self.mutation_rate = self.kwargs_opt["mutation_rate"]
        self.crossover_rate = self.kwargs_opt["crossover_rate"]

        self.max_sample_size = self.kwargs_opt["max_sample_size"]
        self.warm_start_smbo = self.kwargs_opt["warm_start_smbo"]
        self.xi = self.kwargs_opt["xi"]

        if isinstance(self.kwargs_opt["gpr"], str):
            self.gpr = gaussian_process[self.kwargs_opt["gpr"]]
        else:
            self.gpr = self.kwargs_opt["gpr"]

        self.skip_retrain = skip_retrain_[self.kwargs_opt["skip_retrain"]]

        self.start_up_evals = self.kwargs_opt["start_up_evals"]
        self.gamma_tpe = self.kwargs_opt["gamma_tpe"]

        if isinstance(self.kwargs_opt["tree_regressor"], str):
            self.tree_regressor = tree_regressor[self.kwargs_opt["tree_regressor"]]
        else:
            self.tree_regressor = self.kwargs_opt["tree_regressor"]

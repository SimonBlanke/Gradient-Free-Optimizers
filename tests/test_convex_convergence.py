import numpy as np
from tqdm import tqdm

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    TabuOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticTunnelingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    DecisionTreeOptimizer,
)


optimizer_dict = {
    "HillClimbing": HillClimbingOptimizer,
    "StochasticHillClimbing": StochasticHillClimbingOptimizer,
    "TabuSearch": TabuOptimizer,
    "RandomSearch": RandomSearchOptimizer,
    "RandomRestartHillClimbing": RandomRestartHillClimbingOptimizer,
    "RandomAnnealing": RandomAnnealingOptimizer,
    "SimulatedAnnealing": SimulatedAnnealingOptimizer,
    "StochasticTunneling": StochasticTunnelingOptimizer,
    "ParallelTempering": ParallelTemperingOptimizer,
    "ParticleSwarm": ParticleSwarmOptimizer,
    "EvolutionStrategy": EvolutionStrategyOptimizer,
    "Bayesian": BayesianOptimizer,
    "TreeStructured": TreeStructuredParzenEstimators,
    "DecisionTree": DecisionTreeOptimizer,
}


def objective_function(pos_new):
    score = -pos_new[0] * pos_new[0]
    return score


search_space = [np.arange(-100, 100, 1)]
warm_start_ = [[np.array(-100)]]

n_opts = 100
n_iter = 50
min_score_accept = -100


def test_HillClimbingOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = HillClimbingOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_StochasticHillClimbingOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = StochasticHillClimbingOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_TabuOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = TabuOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_RandomSearchOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = RandomSearchOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_RandomRestartHillClimbingOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = RandomRestartHillClimbingOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_RandomAnnealingOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = RandomAnnealingOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_SimulatedAnnealingOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = SimulatedAnnealingOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_StochasticTunnelingOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = StochasticTunnelingOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_ParallelTemperingOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = ParallelTemperingOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_ParticleSwarmOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = ParticleSwarmOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_EvolutionStrategyOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = EvolutionStrategyOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_BayesianOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = BayesianOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_TreeStructuredParzenEstimators_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = TreeStructuredParzenEstimators(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


def test_DecisionTreeOptimizer_convergence():
    scores = []
    for rnd_st in tqdm(range(n_opts)):
        opt = DecisionTreeOptimizer(search_space)
        opt.search(
            objective_function,
            n_iter=n_iter,
            random_state=rnd_st,
            memory=False,
            print_results=False,
            progress_bar=False,
            initialize={"warm_start": warm_start_},
        )

        scores.append(opt.score_best)

    score_mean = np.array(scores).mean()
    assert min_score_accept < score_mean


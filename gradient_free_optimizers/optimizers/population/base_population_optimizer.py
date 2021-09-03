# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..search_tracker import SearchTracker
from ...converter import Converter
from ...results_manager import ResultsManager
from ...optimizers.base_optimizer import get_n_inits


class BasePopulationOptimizer:
    def __init__(
        self,
        search_space,
        initialize={"grid": 4, "random": 2, "vertices": 4},
    ):
        super().__init__()
        self.conv = Converter(search_space)
        self.results_mang = ResultsManager(self.conv)
        self.initialize = initialize

        self.eval_times = []
        self.iter_times = []

    def _iterations(self, positioners):
        nth_iter = 0
        for p in positioners:
            nth_iter = nth_iter + len(p.pos_new_list)

        return nth_iter

    def _create_population(self, Optimizer):
        if isinstance(self.population, int):
            population = []
            for pop_ in range(self.population):
                population.append(
                    Optimizer(self.conv.search_space, rand_rest_p=self.rand_rest_p)
                )
        else:
            population = self.population

        n_inits = get_n_inits(self.initialize)

        if n_inits < len(population):
            print("\n Warning: Not enough initial positions for population size")
            print(" Population size is reduced to", n_inits)
            population = population[:n_inits]

        return population

    def finish_initialization(self):
        pass

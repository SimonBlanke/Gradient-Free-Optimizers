# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .base_population_optimizer import BasePopulationOptimizer



class AntColonyOptimization(BasePopulationOptimizer):
    name = "Ant Colony Optimization"
    _name_ = "ant_colony_optimization"
    __name__ = "AntColonyOptimization"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


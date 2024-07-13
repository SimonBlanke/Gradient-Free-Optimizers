# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer


class EvolutionaryAlgorithmOptimizer(BasePopulationOptimizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

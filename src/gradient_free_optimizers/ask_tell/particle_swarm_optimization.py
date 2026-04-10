# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Particle swarm optimizer with ask/tell interface."""

from typing import Literal

from .._ask_tell_mixin import AskTell
from .._init_utils import get_default_initialize
from ..optimizers import ParticleSwarmOptimizer as _ParticleSwarmOptimizer


class ParticleSwarmOptimizer(_ParticleSwarmOptimizer, AskTell):
    """Particle Swarm optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        Constraint functions restricting the search space.
    random_state : int or None, default=None
        Seed for reproducibility.
    rand_rest_p : float, default=0
        Probability of random restart.
    population : int, default=10
        Number of particles in the swarm.
    inertia : float, default=0.5
        Weight applied to a particle's current velocity.
    cognitive_weight : float, default=0.5
        Attraction strength toward each particle's personal best.
    social_weight : float, default=0.5
        Attraction strength toward the swarm's global best.
    temp_weight : float, default=0.2
        Temperature-like randomness weight added to velocity update.
    """

    def __init__(
        self,
        search_space: dict[str, list],
        initialize: dict[
            Literal["grid", "vertices", "random", "warm_start"],
            int | list[dict],
        ] = None,
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = 10,
        inertia: float = 0.5,
        cognitive_weight: float = 0.5,
        social_weight: float = 0.5,
        temp_weight: float = 0.2,
    ):
        if initialize is None:
            initialize = get_default_initialize()
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            population=population,
            inertia=inertia,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            temp_weight=temp_weight,
        )

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Particle swarm optimizer with ask/tell interface."""

from .._ask_tell_mixin import AskTell
from ..optimizers import ParticleSwarmOptimizer as _ParticleSwarmOptimizer


class ParticleSwarmOptimizer(_ParticleSwarmOptimizer, AskTell):
    """Particle Swarm optimizer with ask/tell interface.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore.
    initial_evaluations : list[tuple[dict, float]]
        Previously evaluated parameters and their scores to seed the optimizer.
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
        initial_evaluations: list[tuple[dict, float]],
        constraints: list[callable] = None,
        random_state: int = None,
        rand_rest_p: float = 0,
        population: int = 10,
        inertia: float = 0.5,
        cognitive_weight: float = 0.5,
        social_weight: float = 0.5,
        temp_weight: float = 0.2,
    ):
        if constraints is None:
            constraints = []

        super().__init__(
            search_space=search_space,
            initialize={"random": 0},
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            population=population,
            inertia=inertia,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            temp_weight=temp_weight,
        )

        self._process_initial_evaluations(initial_evaluations)

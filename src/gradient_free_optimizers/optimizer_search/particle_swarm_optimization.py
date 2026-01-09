# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License
"""Particle swarm optimization inspired by collective swarm behavior."""

from typing import Literal

from .._init_utils import get_default_initialize
from ..optimizers import ParticleSwarmOptimizer as _ParticleSwarmOptimizer
from ..search import Search


class ParticleSwarmOptimizer(_ParticleSwarmOptimizer, Search):
    """
    Swarm intelligence optimizer inspired by collective behavior of bird flocks.

    Particle Swarm Optimization (PSO) is a population-based metaheuristic that
    simulates the social behavior of birds flocking or fish schooling. Each
    particle in the swarm represents a candidate solution that moves through
    the search space influenced by its own best-known position (cognitive
    component) and the swarm's best-known position (social component).

    The velocity update equation balances three components: inertia (tendency
    to continue in the current direction), cognitive attraction (pull toward
    personal best), and social attraction (pull toward global best). This
    creates emergent optimization behavior where particles explore promising
    regions while sharing information about good solutions.

    The algorithm is well-suited for:

    - Continuous optimization problems
    - Multimodal functions where multiple regions need exploration
    - Problems where gradient information is unavailable
    - Real-time optimization where quick convergence is needed

    The balance between `cognitive_weight` and `social_weight` controls
    exploration vs exploitation. Higher cognitive weight promotes individual
    exploration, while higher social weight accelerates convergence toward
    the best-known solution.

    Parameters
    ----------
    search_space : dict[str, list]
        The search space to explore. A dictionary with parameter
        names as keys and a numpy array as values.
    initialize : dict[str, int]
        The method to generate initial positions. A dictionary with
        the following key literals and the corresponding value type:
        {"grid": int, "vertices": int, "random": int, "warm_start": list[dict]}
    constraints : list[callable]
        A list of constraints, where each constraint is a callable.
        The callable returns `True` or `False` dependend on the input parameters.
    random_state : None, int
        If None, create a new random state. If int, create a new random state
        seeded with the value.
    rand_rest_p : float
        The probability of a random iteration during the the search process.
    population : int
        The number of particles in the swarm. Larger populations explore more
        but require more function evaluations per iteration. Default is 10.
    inertia : float
        Weight applied to the particle's current velocity. Controls momentum
        and tendency to continue in the current direction. Values typically
        range from 0.4 to 0.9. Default is 0.5.
    cognitive_weight : float
        Attraction strength toward each particle's personal best position.
        Higher values increase individual exploration. Default is 0.5.
    social_weight : float
        Attraction strength toward the swarm's global best position.
        Higher values increase convergence speed but may cause premature
        convergence. Default is 0.5.
    temp_weight : float
        Temperature-like parameter adding randomness to the velocity update.
        Helps escape local optima. Default is 0.2.

    Examples
    --------
    >>> import numpy as np
    >>> from gradient_free_optimizers import ParticleSwarmOptimizer

    >>> def rastrigin(para):
    ...     x, y = para["x"], para["y"]
    ...     return -(20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

    >>> search_space = {
    ...     "x": np.linspace(-5.12, 5.12, 100),
    ...     "y": np.linspace(-5.12, 5.12, 100),
    ... }

    >>> opt = ParticleSwarmOptimizer(
    ...     search_space, population=20, inertia=0.7,
    ...     cognitive_weight=1.5, social_weight=1.5,
    ... )
    >>> opt.search(rastrigin, n_iter=500)
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
        nth_process: int = None,
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
            nth_process=nth_process,
            population=population,
            inertia=inertia,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            temp_weight=temp_weight,
        )

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Topology functions for population-based optimizers.

Topologies control how individuals in a population share information
by defining each individual's neighborhood. The neighborhood determines
which other individuals can influence a given individual's behavior.

Currently used by ParticleSwarmOptimizer and SpiralOptimization.
"""

from __future__ import annotations

import math
import random

VALID_TOPOLOGIES = ("star", "ring", "von_neumann", "stochastic")
STOCHASTIC_NEIGHBORS = 3


def get_neighbors(topology, particle_idx, population_size):
    """Return neighbor indices for a particle under the given topology.

    The returned list always includes ``particle_idx`` itself.

    Parameters
    ----------
    topology : str
        One of ``"star"``, ``"ring"``, ``"von_neumann"``, ``"stochastic"``.
    particle_idx : int
        Index of the particle in the population.
    population_size : int
        Total number of particles.

    Returns
    -------
    list[int]
        Indices of neighbor particles (always includes particle_idx).

    Raises
    ------
    ValueError
        If ``topology`` is not one of the valid topology names.
    """
    fn = _TOPOLOGY_DISPATCH.get(topology)
    if fn is None:
        raise ValueError(
            f"Unknown topology {topology!r}. Valid topologies: {VALID_TOPOLOGIES}"
        )
    if population_size < 1:
        raise ValueError("population_size must be at least 1")
    if not 0 <= particle_idx < population_size:
        raise ValueError(
            f"particle_idx must be in [0, {population_size}), got {particle_idx}"
        )
    return fn(particle_idx, population_size)


def _unique(indices):
    """Return indices without duplicates while preserving order."""
    result = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            result.append(idx)
            seen.add(idx)
    return result


def _star_neighbors(particle_idx, population_size):
    """All particles are neighbors (fully connected)."""
    return list(range(population_size))


def _ring_neighbors(particle_idx, population_size):
    """Circular ring with one neighbor on each side."""
    if population_size <= 3:
        return list(range(population_size))
    left = (particle_idx - 1) % population_size
    right = (particle_idx + 1) % population_size
    return [particle_idx, left, right]


def _von_neumann_neighbors(particle_idx, population_size):
    """2D toroidal grid with 4 adjacent neighbors (up/down/left/right).

    The population is arranged into an exact rectangular grid (rows * cols
    == population_size). For prime population sizes the grid degenerates
    to a single row or column, which produces ring-like neighborhoods.
    """
    if population_size <= 5:
        return list(range(population_size))
    rows, cols = _best_grid(population_size)
    r, c = divmod(particle_idx, cols)
    return _unique(
        [
            particle_idx,
            ((r - 1) % rows) * cols + c,
            ((r + 1) % rows) * cols + c,
            r * cols + (c - 1) % cols,
            r * cols + (c + 1) % cols,
        ]
    )


def _stochastic_neighbors(particle_idx, population_size, k=STOCHASTIC_NEIGHBORS):
    """Random neighborhood of k particles, resampled each call."""
    if population_size <= k + 1:
        return list(range(population_size))
    others = [i for i in range(population_size) if i != particle_idx]
    selected = random.sample(others, k)
    return [particle_idx] + selected


def _best_grid(n):
    """Find rows x cols where rows * cols == n, closest to square.

    Always returns an exact factorization. For primes the result is
    (n, 1) or (1, n), giving a degenerate single-row or single-column grid.
    """
    best_cols = 1
    best_rows = n
    for c in range(2, int(math.isqrt(n)) + 1):
        if n % c == 0:
            best_cols = c
            best_rows = n // c
    return best_rows, best_cols


_TOPOLOGY_DISPATCH = {
    "star": _star_neighbors,
    "ring": _ring_neighbors,
    "von_neumann": _von_neumann_neighbors,
    "stochastic": _stochastic_neighbors,
}

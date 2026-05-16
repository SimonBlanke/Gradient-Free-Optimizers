"""Regression tests for PSO and Spiral optimizer state updates."""

import numpy as np
import pytest

from gradient_free_optimizers import ParticleSwarmOptimizer, SpiralOptimization
from gradient_free_optimizers.optimizers.pop_opt._spiral import rotation


def objective_function(para):
    return -(para["x"] * para["x"])


def test_pso_updates_particle_current_and_personal_best_every_evaluation():
    search_space = {"x": np.arange(-10, 11)}
    opt = ParticleSwarmOptimizer(
        search_space,
        initialize={"warm_start": [{"x": 10}, {"x": 0}]},
        population=2,
        random_state=1,
        inertia=0.0,
        cognitive_weight=0.0,
        social_weight=2.0,
    )

    opt.search(objective_function, n_iter=3, memory=False, verbosity=False)

    particle = opt.particles[0]
    assert list(opt.conv.position2value(particle._pos_current)) == [8]
    assert particle._score_current == -64
    assert list(opt.conv.position2value(particle._pos_best)) == [8]
    assert particle._score_best == -64


def test_pso_sorts_particles_by_personal_best_score():
    search_space = {"x": np.arange(-10, 11)}
    opt = ParticleSwarmOptimizer(
        search_space,
        initialize={"warm_start": [{"x": -10}, {"x": 0}]},
        population=2,
        random_state=0,
    )
    first, second = opt.particles

    first._pos_current = np.array([0])
    first._score_current = -100
    first._pos_best = np.array([0])
    first._score_best = 10

    second._pos_current = np.array([1])
    second._score_current = 100
    second._pos_best = np.array([1])
    second._score_best = -10

    opt._sort_pop_personal_best_score()

    assert opt.pop_sorted[0] is first


def test_spiral_1d_rotation_flips_vector_sign():
    assert list(rotation(1, np.array([5.0]))) == [-5.0]


def test_spiral_2d_rotation_degrees_controls_angle():
    rotated = rotation(2, np.array([1.0, 0.0]), rotation_degrees=45.0)

    assert list(rotated) == pytest.approx([np.sqrt(0.5), np.sqrt(0.5)])


def test_spiral_high_dimensional_rotation_preserves_norm_and_angle():
    vector = np.array([1.0, 1.0, 1.0])
    rotated = np.array(rotation(3, vector, rotation_degrees=90.0))

    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(vector))
    assert np.dot(vector, rotated) == pytest.approx(0.0)


def test_spiral_movement_uses_normalized_coordinates():
    unit_space = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    wide_space = {"x": (-100.0, 100.0), "y": (-100.0, 100.0)}

    unit_opt = SpiralOptimization(
        unit_space,
        initialize={"warm_start": [{"x": 0.0, "y": 0.0}]},
        population=1,
        decay_rate=1.0,
        spiral_radius=1.0,
    )
    wide_opt = SpiralOptimization(
        wide_space,
        initialize={"warm_start": [{"x": 0.0, "y": 0.0}]},
        population=1,
        decay_rate=1.0,
        spiral_radius=1.0,
    )

    unit_opt.center_pos = np.array([0.0, 0.0])
    unit_opt.p_current = unit_opt.particles[0]
    unit_opt.p_current._pos_current = np.array([1.0, 0.0])

    wide_opt.center_pos = np.array([0.0, 0.0])
    wide_opt.p_current = wide_opt.particles[0]
    wide_opt.p_current._pos_current = np.array([100.0, 0.0])

    assert list(unit_opt._compute_spiral_position()) == [0.0, 1.0]
    assert list(wide_opt._compute_spiral_position()) == [0.0, 100.0]

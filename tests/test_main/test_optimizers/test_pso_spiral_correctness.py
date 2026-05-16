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


def test_pso_temp_weight_adds_velocity_vibration():
    search_space = {"x": (-10.0, 10.0)}

    def prepared_optimizer(temp_weight):
        opt = ParticleSwarmOptimizer(
            search_space,
            initialize={"warm_start": [{"x": 0.0}]},
            population=1,
            random_state=3,
            inertia=0.0,
            cognitive_weight=0.0,
            social_weight=0.0,
            temp_weight=temp_weight,
        )
        particle = opt.particles[0]
        position = np.array([0.0])
        particle._pos_current = position.copy()
        particle._pos_best = position.copy()
        particle.global_pos_best = position.copy()
        particle.velo = position.copy()
        opt.p_current = particle
        return opt

    no_vibration = prepared_optimizer(temp_weight=0.0)._compute_pso_position()
    vibration = prepared_optimizer(temp_weight=1.0)._compute_pso_position()

    assert list(no_vibration) == [0.0]
    assert vibration[0] != pytest.approx(0.0)
    assert -1.0 <= vibration[0] <= 1.0


def test_pso_constraint_retry_regenerates_candidate_and_rolls_back_velocity():
    opt = ParticleSwarmOptimizer(
        {"x": (-10.0, 10.0)},
        initialize={"warm_start": [{"x": 0.0}]},
        population=1,
        random_state=3,
        inertia=1.0,
        cognitive_weight=0.0,
        social_weight=0.0,
        temp_weight=1.0,
    )
    particle = opt.particles[0]
    position = np.array([0.0])
    particle._pos_current = position.copy()
    particle._pos_best = position.copy()
    particle._score_current = 0.0
    particle._score_best = 0.0
    particle.global_pos_best = position.copy()
    particle.velo = position.copy()

    vibrations = [np.array([-1.0]), np.array([1.0])]
    opt._compute_temperature_vibration = lambda n_dims: vibrations.pop(0)

    checked_values = []

    def not_in_constraint(candidate):
        value = float(opt.conv.position2value(candidate)[0])
        checked_values.append(value)
        return value > 0.5

    opt.conv.not_in_constraint = not_in_constraint
    candidate = opt._iterate()

    assert checked_values == [-1.0, 1.0]
    assert list(candidate) == [1.0]
    assert list(particle.velo) == [1.0]


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


def test_spiral_updates_center_after_evaluating_new_best():
    search_space = {"x": np.arange(-10, 11)}
    opt = SpiralOptimization(
        search_space,
        initialize={"warm_start": [{"x": 10}, {"x": 5}]},
        population=2,
        random_state=0,
        decay_rate=1.0,
        spiral_radius=1.0,
    )

    opt.search(objective_function, n_iter=3, memory=False, verbosity=False)

    assert opt.best_para == {"x": 0}
    assert opt.best_score == 0.0
    assert list(opt.conv.position2value(opt.center_pos)) == [0]
    assert opt.center_score == 0.0


def test_spiral_constraint_retry_regenerates_with_contracted_decay():
    opt = SpiralOptimization(
        {"x": (-10.0, 10.0)},
        initialize={"warm_start": [{"x": 10.0}]},
        population=1,
        random_state=0,
        decay_rate=0.5,
        spiral_radius=1.0,
    )
    particle = opt.particles[0]
    particle._pos_current = np.array([10.0])
    particle._pos_best = np.array([0.0])
    particle._score_current = 0.0
    particle._score_best = 0.0
    opt.center_pos = np.array([0.0])
    opt.center_score = 0.0
    opt._decay_factor = 1.0

    checked_values = []

    def not_in_constraint(candidate):
        value = float(opt.conv.position2value(candidate)[0])
        checked_values.append(value)
        return value > -3.0

    opt.conv.not_in_constraint = not_in_constraint
    candidate = opt._iterate()

    assert checked_values == [-5.0, -2.5]
    assert list(candidate) == [-2.5]
    assert opt._decay_factor == 0.25


def test_spiral_constraint_random_fallback_does_not_commit_retry_decay():
    opt = SpiralOptimization(
        {"x": (-10.0, 10.0)},
        initialize={"warm_start": [{"x": 10.0}]},
        population=1,
        random_state=0,
        decay_rate=0.5,
        spiral_radius=1.0,
    )
    particle = opt.particles[0]
    particle._pos_current = np.array([10.0])
    particle._pos_best = np.array([0.0])
    particle._score_current = 0.0
    particle._score_best = 0.0
    opt.center_pos = np.array([0.0])
    opt.center_score = 0.0
    opt._decay_factor = 1.0
    opt.init.move_random_typed = lambda: np.array([7.0])

    def not_in_constraint(candidate):
        value = float(opt.conv.position2value(candidate)[0])
        return value == 7.0

    opt.conv.not_in_constraint = not_in_constraint
    candidate = opt._iterate()

    assert list(candidate) == [7.0]
    assert opt._decay_factor == 1.0

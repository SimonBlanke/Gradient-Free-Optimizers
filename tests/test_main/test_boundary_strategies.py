"""Tests for optimizer boundary handling strategies."""

import inspect

import numpy as np
import pytest

from gradient_free_optimizers import ask_tell, optimizer_search
from gradient_free_optimizers.ask_tell import __all__ as ask_tell_names
from gradient_free_optimizers.optimizers.global_opt.pattern_search import PatternSearch
from gradient_free_optimizers.optimizers.global_opt.powells_method import PowellsMethod
from gradient_free_optimizers.optimizers.local_opt import HillClimbingOptimizer
from gradient_free_optimizers.optimizers.local_opt.downhill_simplex import (
    DownhillSimplexOptimizer,
)
from gradient_free_optimizers.optimizers.pop_opt.differential_evolution import (
    DifferentialEvolutionOptimizer,
)

BOUNDARIES = ("clip", "reflect", "periodic", "random", "intermediate")


@pytest.fixture
def mixed_search_space():
    return {
        "x": (0.0, 10.0),
        "cat": ["a", "b", "c"],
        "d": np.array([10, 20, 30, 40, 50]),
    }


def make_optimizer(mixed_search_space, boundary):
    return HillClimbingOptimizer(
        mixed_search_space,
        random_state=123,
        boundary=boundary,
    )


def test_invalid_boundary_raises(mixed_search_space):
    with pytest.raises(ValueError, match="boundary must be one of"):
        make_optimizer(mixed_search_space, "missing")


def test_default_boundary_is_clip(mixed_search_space):
    opt = HillClimbingOptimizer(mixed_search_space, random_state=123)

    clipped = opt._clip_position(np.array([-2.0, 20.0, 8.0]))

    assert np.allclose(clipped, np.array([0.0, 2.0, 4.0]))


def test_categorical_dimensions_always_clip(mixed_search_space):
    for boundary in BOUNDARIES:
        opt = make_optimizer(mixed_search_space, boundary)
        opt._pos_current = np.array([5.0, 1.0, 2.0])

        clipped = opt._clip_position(np.array([5.0, 20.0, 2.0]))

        assert clipped[1] == 2


def test_reflect_boundary_for_continuous_and_discrete(mixed_search_space):
    opt = make_optimizer(mixed_search_space, "reflect")

    clipped_above = opt._clip_position(np.array([12.0, 1.0, 5.0]))
    clipped_below = opt._clip_position(np.array([-2.0, 1.0, -1.0]))

    assert np.allclose(clipped_above, np.array([8.0, 1.0, 3.0]))
    assert np.allclose(clipped_below, np.array([2.0, 1.0, 1.0]))


def test_periodic_boundary_for_continuous_and_discrete(mixed_search_space):
    opt = make_optimizer(mixed_search_space, "periodic")

    clipped_above = opt._clip_position(np.array([12.0, 1.0, 5.0]))
    clipped_below = opt._clip_position(np.array([-2.0, 1.0, -1.0]))
    clipped_at_max = opt._clip_position(np.array([10.0, 1.0, 4.0]))

    assert np.allclose(clipped_above, np.array([2.0, 1.0, 0.0]))
    assert np.allclose(clipped_below, np.array([8.0, 1.0, 4.0]))
    assert np.allclose(clipped_at_max, np.array([10.0, 1.0, 4.0]))


def test_random_boundary_replaces_only_out_of_bounds_values(mixed_search_space):
    opt = make_optimizer(mixed_search_space, "random")

    in_bounds = opt._clip_position(np.array([5.0, 1.0, 2.0]))
    clipped = opt._clip_position(np.array([12.0, 1.0, 5.0]))

    assert np.allclose(in_bounds, np.array([5.0, 1.0, 2.0]))
    assert 0.0 <= clipped[0] <= 10.0
    assert clipped[1] == 1
    assert clipped[2] in {0, 1, 2, 3, 4}


def test_random_boundary_is_reproducible(mixed_search_space):
    opt_1 = make_optimizer(mixed_search_space, "random")
    opt_2 = make_optimizer(mixed_search_space, "random")

    clipped_1 = opt_1._clip_position(np.array([12.0, 1.0, 5.0]))
    clipped_2 = opt_2._clip_position(np.array([12.0, 1.0, 5.0]))

    assert np.allclose(clipped_1, clipped_2)


def test_intermediate_boundary_uses_current_position(mixed_search_space):
    opt = make_optimizer(mixed_search_space, "intermediate")
    opt._pos_current = np.array([6.0, 1.0, 2.0])

    clipped_above = opt._clip_position(np.array([12.0, 1.0, 5.0]))
    clipped_below = opt._clip_position(np.array([-2.0, 1.0, -1.0]))

    assert np.allclose(clipped_above, np.array([8.0, 1.0, 3.0]))
    assert np.allclose(clipped_below, np.array([3.0, 1.0, 1.0]))


def test_intermediate_boundary_falls_back_to_clip_without_current(mixed_search_space):
    opt = make_optimizer(mixed_search_space, "intermediate")

    clipped = opt._clip_position(np.array([12.0, 1.0, 5.0]))

    assert np.allclose(clipped, np.array([10.0, 1.0, 4.0]))


@pytest.mark.parametrize(
    ("optimizer_cls", "helper_name"),
    [
        (PatternSearch, "_clip_to_bounds"),
        (DownhillSimplexOptimizer, "_clip_to_bounds"),
        (PowellsMethod, "_conv2pos_typed"),
        (DifferentialEvolutionOptimizer, "_conv2pos_typed"),
    ],
)
def test_internal_boundary_helpers_use_configured_strategy(
    mixed_search_space, optimizer_cls, helper_name
):
    opt = optimizer_cls(mixed_search_space, random_state=123, boundary="periodic")
    helper = getattr(opt, helper_name)

    clipped = helper(np.array([12.0, 1.0, 5.0]))

    assert np.allclose(clipped, np.array([2.0, 1.0, 0.0]))


def test_optimizer_search_classes_expose_boundary():
    for name in optimizer_search.__all__:
        signature = inspect.signature(getattr(optimizer_search, name))
        assert "boundary" in signature.parameters, name


def test_ask_tell_classes_expose_boundary():
    for name in ask_tell_names:
        signature = inspect.signature(getattr(ask_tell, name))
        assert "boundary" in signature.parameters, name

"""Tests for constrained optimization."""

import numpy as np
import pytest

from ._parametrize import optimizers


@pytest.mark.parametrize(*optimizers)
def test_constraint_single(Optimizer):
    search_space = {"x1": np.arange(-10, 10, 1)}
    opt = Optimizer(
        search_space,
        constraints=[lambda p: p["x1"] > -5],
        random_state=42,
    )
    opt.search(lambda p: -(p["x1"] ** 2), n_iter=30, verbosity=False)

    values = opt.search_data["x1"].values
    assert np.all(values > -5)


@pytest.mark.parametrize(*optimizers)
def test_constraint_multiple(Optimizer):
    search_space = {"x1": np.arange(-10, 10, 0.5)}
    opt = Optimizer(
        search_space,
        constraints=[lambda p: p["x1"] > -3, lambda p: p["x1"] < 3],
        random_state=42,
    )
    opt.search(lambda p: -(p["x1"] ** 2), n_iter=30, verbosity=False)

    values = opt.search_data["x1"].values
    assert np.all(values > -3)
    assert np.all(values < 3)


@pytest.mark.parametrize(*optimizers)
def test_constraint_2d(Optimizer):
    search_space = {
        "x1": np.arange(-10, 10, 1),
        "x2": np.arange(-10, 10, 1),
    }
    opt = Optimizer(
        search_space,
        constraints=[lambda p: p["x1"] > 0, lambda p: p["x2"] > 0],
        random_state=42,
    )
    opt.search(
        lambda p: -(p["x1"] ** 2 + p["x2"] ** 2),
        n_iter=50,
        verbosity=False,
    )

    data = opt.search_data
    assert np.all(data["x1"].values > 0)
    assert np.all(data["x2"].values > 0)


@pytest.mark.parametrize(*optimizers)
def test_constraint_adversarial(Optimizer):
    """Optimum at x1=-80 is in the excluded region (x1 <= 50)."""
    search_space = {"x1": np.arange(-100, 100, 1)}
    opt = Optimizer(
        search_space,
        constraints=[lambda p: p["x1"] > 50],
        random_state=42,
    )
    opt.search(
        lambda p: -abs(p["x1"] - (-80)),
        n_iter=100,
        verbosity=False,
    )

    values = opt.search_data["x1"].values
    assert np.all(values > 50)


@pytest.mark.parametrize(*optimizers)
def test_constraint_tracking_consistency(Optimizer):
    n_iter = 30
    search_space = {"x1": np.arange(-10, 10, 0.1)}

    opt = Optimizer(
        search_space,
        constraints=[lambda p: p["x1"] > -5, lambda p: p["x1"] < 5],
        random_state=42,
    )
    opt.search(lambda p: -(p["x1"] ** 2), n_iter=n_iter, verbosity=False)

    n_new = sum(len(o._pos_new_list) for o in opt.optimizers)
    n_new_scores = sum(len(o._score_new_list) for o in opt.optimizers)
    n_current = sum(len(o._pos_current_list) for o in opt.optimizers)
    n_current_scores = sum(len(o._score_current_list) for o in opt.optimizers)
    n_best = sum(len(o._pos_best_list) for o in opt.optimizers)
    n_best_scores = sum(len(o._score_best_list) for o in opt.optimizers)

    assert n_new == n_iter
    assert n_new == n_new_scores
    assert n_current == n_current_scores
    assert n_current <= n_new
    assert n_best == n_best_scores
    assert n_best <= n_new

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API Freeze Tests for Global Optimizers.

These tests ensure that the public API of global optimizers remains stable.
Any changes to parameter names, default values, or method signatures will
cause these tests to fail, alerting developers to potential breaking changes.

Global optimizers covered:
- RandomSearchOptimizer
- GridSearchOptimizer
- RandomRestartHillClimbingOptimizer
- PowellsMethod
- PatternSearch
"""

import numpy as np
import pytest

from gradient_free_optimizers import (
    GridSearchOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
)

# Minimal search space and objective for smoke tests
SEARCH_SPACE = {"x": np.linspace(-1, 1, 10)}


def objective(p):
    return -(p["x"] ** 2)


# =============================================================================
# Base parameter tests (common to all optimizers)
# =============================================================================


@pytest.mark.parametrize(
    "Optimizer",
    [
        RandomSearchOptimizer,
        GridSearchOptimizer,
        RandomRestartHillClimbingOptimizer,
        PowellsMethod,
        PatternSearch,
    ],
)
class TestBaseParameters:
    """Test base parameters common to all global optimizers."""

    def test_search_space_required(self, Optimizer):
        """Verify search_space is a required parameter."""
        opt = Optimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_initialize_parameter(self, Optimizer):
        """Test initialize parameter accepts dict."""
        opt = Optimizer(SEARCH_SPACE, initialize={"random": 2})
        opt.search(objective, n_iter=1, verbosity=False)

    def test_constraints_parameter(self, Optimizer):
        """Test constraints parameter accepts list."""
        opt = Optimizer(SEARCH_SPACE, constraints=[])
        opt.search(objective, n_iter=1, verbosity=False)

    def test_random_state_parameter(self, Optimizer):
        """Test random_state parameter accepts int."""
        opt = Optimizer(SEARCH_SPACE, random_state=42)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_nth_process_parameter(self, Optimizer):
        """Test nth_process parameter accepts int."""
        opt = Optimizer(SEARCH_SPACE, nth_process=0)
        opt.search(objective, n_iter=1, verbosity=False)


@pytest.mark.parametrize(
    "Optimizer",
    [
        GridSearchOptimizer,
        RandomRestartHillClimbingOptimizer,
        PowellsMethod,
        PatternSearch,
    ],
)
class TestRandRestPParameter:
    """Test rand_rest_p for optimizers that support it (not RandomSearchOptimizer)."""

    def test_rand_rest_p_parameter(self, Optimizer):
        """Test rand_rest_p parameter accepts float."""
        opt = Optimizer(SEARCH_SPACE, rand_rest_p=0.1)
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# RandomSearchOptimizer API Tests
# =============================================================================


class TestRandomSearchOptimizerAPI:
    """API freeze tests for RandomSearchOptimizer.

    Note: RandomSearchOptimizer does not support rand_rest_p in its public API.
    """

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = RandomSearchOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = RandomSearchOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            nth_process=None,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# GridSearchOptimizer API Tests
# =============================================================================


class TestGridSearchOptimizerAPI:
    """API freeze tests for GridSearchOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = GridSearchOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_step_size_parameter(self):
        """Test step_size parameter exists and accepts int."""
        opt = GridSearchOptimizer(SEARCH_SPACE, step_size=2)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_direction_diagonal(self):
        """Test direction parameter accepts 'diagonal'."""
        opt = GridSearchOptimizer(SEARCH_SPACE, direction="diagonal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_direction_orthogonal(self):
        """Test direction parameter accepts 'orthogonal'."""
        opt = GridSearchOptimizer(SEARCH_SPACE, direction="orthogonal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = GridSearchOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            step_size=1,
            direction="diagonal",
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# RandomRestartHillClimbingOptimizer API Tests
# =============================================================================


class TestRandomRestartHillClimbingOptimizerAPI:
    """API freeze tests for RandomRestartHillClimbingOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = RandomRestartHillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_epsilon_parameter(self):
        """Test epsilon parameter exists (inherited from HC)."""
        opt = RandomRestartHillClimbingOptimizer(SEARCH_SPACE, epsilon=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_parameter(self):
        """Test distribution parameter exists (inherited from HC)."""
        opt = RandomRestartHillClimbingOptimizer(SEARCH_SPACE, distribution="normal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_neighbours_parameter(self):
        """Test n_neighbours parameter exists (inherited from HC)."""
        opt = RandomRestartHillClimbingOptimizer(SEARCH_SPACE, n_neighbours=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_iter_restart_parameter(self):
        """Test n_iter_restart parameter exists and accepts int."""
        opt = RandomRestartHillClimbingOptimizer(SEARCH_SPACE, n_iter_restart=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = RandomRestartHillClimbingOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
            n_iter_restart=10,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# PowellsMethod API Tests
# =============================================================================


class TestPowellsMethodAPI:
    """API freeze tests for PowellsMethod."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = PowellsMethod(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_epsilon_parameter(self):
        """Test epsilon parameter exists (inherited from HC)."""
        opt = PowellsMethod(SEARCH_SPACE, epsilon=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_parameter(self):
        """Test distribution parameter exists (inherited from HC)."""
        opt = PowellsMethod(SEARCH_SPACE, distribution="normal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_neighbours_parameter(self):
        """Test n_neighbours parameter exists (inherited from HC)."""
        opt = PowellsMethod(SEARCH_SPACE, n_neighbours=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_iters_p_dim_parameter(self):
        """Test iters_p_dim parameter exists and accepts int."""
        opt = PowellsMethod(SEARCH_SPACE, iters_p_dim=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_line_search_grid(self):
        """Test line_search parameter accepts 'grid'."""
        opt = PowellsMethod(SEARCH_SPACE, line_search="grid")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_line_search_golden(self):
        """Test line_search parameter accepts 'golden'."""
        opt = PowellsMethod(SEARCH_SPACE, line_search="golden")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_line_search_hill_climb(self):
        """Test line_search parameter accepts 'hill_climb'."""
        opt = PowellsMethod(SEARCH_SPACE, line_search="hill_climb")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_convergence_threshold_parameter(self):
        """Test convergence_threshold parameter exists and accepts float."""
        opt = PowellsMethod(SEARCH_SPACE, convergence_threshold=1e-6)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = PowellsMethod(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
            iters_p_dim=10,
            line_search="grid",
            convergence_threshold=1e-8,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# PatternSearch API Tests
# =============================================================================


class TestPatternSearchAPI:
    """API freeze tests for PatternSearch."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = PatternSearch(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_positions_parameter(self):
        """Test n_positions parameter exists and accepts int."""
        opt = PatternSearch(SEARCH_SPACE, n_positions=6)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_pattern_size_parameter(self):
        """Test pattern_size parameter exists and accepts float."""
        opt = PatternSearch(SEARCH_SPACE, pattern_size=0.5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_reduction_parameter(self):
        """Test reduction parameter exists and accepts float."""
        opt = PatternSearch(SEARCH_SPACE, reduction=0.8)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = PatternSearch(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            n_positions=4,
            pattern_size=0.25,
            reduction=0.9,
        )
        opt.search(objective, n_iter=1, verbosity=False)

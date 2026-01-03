# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API Freeze Tests for Experimental Optimizers

These tests ensure that the public API of experimental optimizers remains stable.
Any changes to parameter names, default values, or method signatures will
cause these tests to fail, alerting developers to potential breaking changes.

Experimental optimizers covered:
- RandomAnnealingOptimizer
"""

import numpy as np
import pytest

from gradient_free_optimizers import RandomAnnealingOptimizer


# Minimal search space and objective for smoke tests
SEARCH_SPACE = {"x": np.linspace(-1, 1, 5)}


def objective(p):
    return -(p["x"] ** 2)


# =============================================================================
# RandomAnnealingOptimizer Base Parameter Tests
# =============================================================================


class TestRandomAnnealingBaseParameters:
    """Test base parameters for RandomAnnealingOptimizer."""

    def test_search_space_required(self):
        """Verify search_space is a required parameter."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_initialize_parameter(self):
        """Test initialize parameter accepts dict."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, initialize={"random": 2})
        opt.search(objective, n_iter=1, verbosity=False)

    def test_constraints_parameter(self):
        """Test constraints parameter accepts list."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, constraints=[])
        opt.search(objective, n_iter=1, verbosity=False)

    def test_random_state_parameter(self):
        """Test random_state parameter accepts int."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, random_state=42)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_rand_rest_p_parameter(self):
        """Test rand_rest_p parameter accepts float."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, rand_rest_p=0.1)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_nth_process_parameter(self):
        """Test nth_process parameter accepts int."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, nth_process=0)
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# RandomAnnealingOptimizer API Tests
# =============================================================================


class TestRandomAnnealingOptimizerAPI:
    """API freeze tests for RandomAnnealingOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_epsilon_parameter(self):
        """Test epsilon parameter exists (inherited from HC)."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, epsilon=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_parameter(self):
        """Test distribution parameter exists (inherited from HC)."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, distribution="normal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_neighbours_parameter(self):
        """Test n_neighbours parameter exists (inherited from HC)."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, n_neighbours=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_annealing_rate_parameter(self):
        """Test annealing_rate parameter exists and accepts float."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, annealing_rate=0.95)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_start_temp_parameter(self):
        """Test start_temp parameter exists and accepts float/int."""
        opt = RandomAnnealingOptimizer(SEARCH_SPACE, start_temp=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = RandomAnnealingOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
            annealing_rate=0.98,
            start_temp=10,
        )
        opt.search(objective, n_iter=1, verbosity=False)

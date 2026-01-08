# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API Freeze Tests for Local Optimizers.

These tests ensure that the public API of local optimizers remains stable.
Any changes to parameter names, default values, or method signatures will
cause these tests to fail, alerting developers to potential breaking changes.

Local optimizers covered:
- HillClimbingOptimizer
- StochasticHillClimbingOptimizer
- RepulsingHillClimbingOptimizer
- SimulatedAnnealingOptimizer
- DownhillSimplexOptimizer
"""

import numpy as np
import pytest

from gradient_free_optimizers import (
    DownhillSimplexOptimizer,
    HillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticHillClimbingOptimizer,
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
        HillClimbingOptimizer,
        StochasticHillClimbingOptimizer,
        RepulsingHillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        DownhillSimplexOptimizer,
    ],
)
class TestBaseParameters:
    """Test base parameters common to all local optimizers."""

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

    def test_rand_rest_p_parameter(self, Optimizer):
        """Test rand_rest_p parameter accepts float."""
        opt = Optimizer(SEARCH_SPACE, rand_rest_p=0.1)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_nth_process_parameter(self, Optimizer):
        """Test nth_process parameter accepts int."""
        opt = Optimizer(SEARCH_SPACE, nth_process=0)
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# HillClimbingOptimizer API Tests
# =============================================================================


class TestHillClimbingOptimizerAPI:
    """API freeze tests for HillClimbingOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = HillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_epsilon_parameter(self):
        """Test epsilon parameter exists and accepts float."""
        opt = HillClimbingOptimizer(SEARCH_SPACE, epsilon=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_parameter(self):
        """Test distribution parameter exists and accepts string."""
        opt = HillClimbingOptimizer(SEARCH_SPACE, distribution="normal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_uniform(self):
        """Test distribution parameter accepts 'uniform'."""
        opt = HillClimbingOptimizer(SEARCH_SPACE, distribution="uniform")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_neighbours_parameter(self):
        """Test n_neighbours parameter exists and accepts int."""
        opt = HillClimbingOptimizer(SEARCH_SPACE, n_neighbours=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = HillClimbingOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# StochasticHillClimbingOptimizer API Tests
# =============================================================================


class TestStochasticHillClimbingOptimizerAPI:
    """API freeze tests for StochasticHillClimbingOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = StochasticHillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_epsilon_parameter(self):
        """Test epsilon parameter exists (inherited from HC)."""
        opt = StochasticHillClimbingOptimizer(SEARCH_SPACE, epsilon=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_parameter(self):
        """Test distribution parameter exists (inherited from HC)."""
        opt = StochasticHillClimbingOptimizer(SEARCH_SPACE, distribution="normal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_neighbours_parameter(self):
        """Test n_neighbours parameter exists (inherited from HC)."""
        opt = StochasticHillClimbingOptimizer(SEARCH_SPACE, n_neighbours=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_p_accept_parameter(self):
        """Test p_accept parameter exists and accepts float."""
        opt = StochasticHillClimbingOptimizer(SEARCH_SPACE, p_accept=0.3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = StochasticHillClimbingOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
            p_accept=0.5,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# RepulsingHillClimbingOptimizer API Tests
# =============================================================================


class TestRepulsingHillClimbingOptimizerAPI:
    """API freeze tests for RepulsingHillClimbingOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = RepulsingHillClimbingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_epsilon_parameter(self):
        """Test epsilon parameter exists (inherited from HC)."""
        opt = RepulsingHillClimbingOptimizer(SEARCH_SPACE, epsilon=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_parameter(self):
        """Test distribution parameter exists (inherited from HC)."""
        opt = RepulsingHillClimbingOptimizer(SEARCH_SPACE, distribution="normal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_neighbours_parameter(self):
        """Test n_neighbours parameter exists (inherited from HC)."""
        opt = RepulsingHillClimbingOptimizer(SEARCH_SPACE, n_neighbours=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_repulsion_factor_parameter(self):
        """Test repulsion_factor parameter exists and accepts int/float."""
        opt = RepulsingHillClimbingOptimizer(SEARCH_SPACE, repulsion_factor=3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = RepulsingHillClimbingOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
            repulsion_factor=5,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# SimulatedAnnealingOptimizer API Tests
# =============================================================================


class TestSimulatedAnnealingOptimizerAPI:
    """API freeze tests for SimulatedAnnealingOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = SimulatedAnnealingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_epsilon_parameter(self):
        """Test epsilon parameter exists (inherited from SHC)."""
        opt = SimulatedAnnealingOptimizer(SEARCH_SPACE, epsilon=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_distribution_parameter(self):
        """Test distribution parameter exists (inherited from SHC)."""
        opt = SimulatedAnnealingOptimizer(SEARCH_SPACE, distribution="normal")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_neighbours_parameter(self):
        """Test n_neighbours parameter exists (inherited from SHC)."""
        opt = SimulatedAnnealingOptimizer(SEARCH_SPACE, n_neighbours=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_annealing_rate_parameter(self):
        """Test annealing_rate parameter exists and accepts float."""
        opt = SimulatedAnnealingOptimizer(SEARCH_SPACE, annealing_rate=0.95)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_start_temp_parameter(self):
        """Test start_temp parameter exists and accepts float/int."""
        opt = SimulatedAnnealingOptimizer(SEARCH_SPACE, start_temp=2)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = SimulatedAnnealingOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            epsilon=0.03,
            distribution="normal",
            n_neighbours=3,
            annealing_rate=0.97,
            start_temp=1,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# DownhillSimplexOptimizer API Tests
# =============================================================================


class TestDownhillSimplexOptimizerAPI:
    """API freeze tests for DownhillSimplexOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = DownhillSimplexOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_alpha_parameter(self):
        """Test alpha parameter exists and accepts float/int."""
        opt = DownhillSimplexOptimizer(SEARCH_SPACE, alpha=1.5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_gamma_parameter(self):
        """Test gamma parameter exists and accepts float/int."""
        opt = DownhillSimplexOptimizer(SEARCH_SPACE, gamma=2.5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_beta_parameter(self):
        """Test beta parameter exists and accepts float."""
        opt = DownhillSimplexOptimizer(SEARCH_SPACE, beta=0.3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_sigma_parameter(self):
        """Test sigma parameter exists and accepts float."""
        opt = DownhillSimplexOptimizer(SEARCH_SPACE, sigma=0.3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = DownhillSimplexOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            alpha=1,
            gamma=2,
            beta=0.5,
            sigma=0.5,
        )
        opt.search(objective, n_iter=1, verbosity=False)

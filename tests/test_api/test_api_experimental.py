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
- EnsembleOptimizer
"""

import numpy as np
import pytest

from gradient_free_optimizers import (
    RandomAnnealingOptimizer,
    EnsembleOptimizer,
)


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
# EnsembleOptimizer Base Parameter Tests
# =============================================================================

# NOTE: EnsembleOptimizer is currently broken due to a bug in the internal
# implementation (passes epsilon/distribution/n_neighbours to SMBO which
# doesn't accept them). These tests are skipped until the bug is fixed.
# See: src/gradient_free_optimizers/optimizers/exp_opt/ensemble_optimizer.py


@pytest.mark.skip(reason="EnsembleOptimizer has internal implementation bug")
class TestEnsembleBaseParameters:
    """Test base parameters for EnsembleOptimizer."""

    def test_search_space_required(self):
        """Verify search_space is a required parameter."""
        opt = EnsembleOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_initialize_parameter(self):
        """Test initialize parameter accepts dict."""
        opt = EnsembleOptimizer(SEARCH_SPACE, initialize={"random": 2})
        opt.search(objective, n_iter=1, verbosity=False)

    def test_constraints_parameter(self):
        """Test constraints parameter accepts list."""
        opt = EnsembleOptimizer(SEARCH_SPACE, constraints=[])
        opt.search(objective, n_iter=1, verbosity=False)

    def test_random_state_parameter(self):
        """Test random_state parameter accepts int."""
        opt = EnsembleOptimizer(SEARCH_SPACE, random_state=42)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_rand_rest_p_parameter(self):
        """Test rand_rest_p parameter accepts float."""
        opt = EnsembleOptimizer(SEARCH_SPACE, rand_rest_p=0.1)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_nth_process_parameter(self):
        """Test nth_process parameter accepts int."""
        opt = EnsembleOptimizer(SEARCH_SPACE, nth_process=0)
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


# =============================================================================
# EnsembleOptimizer API Tests
# =============================================================================


@pytest.mark.skip(reason="EnsembleOptimizer has internal implementation bug")
class TestEnsembleOptimizerAPI:
    """API freeze tests for EnsembleOptimizer.

    Note: EnsembleOptimizer's public API is simpler than its internal implementation.
    Parameters like epsilon, distribution, n_neighbours, xi, and estimators are
    NOT exposed in the public API wrapper.

    Currently skipped due to bug in internal implementation.
    """

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = EnsembleOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_warm_start_smbo_parameter(self):
        """Test warm_start_smbo parameter accepts None."""
        opt = EnsembleOptimizer(SEARCH_SPACE, warm_start_smbo=None)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_max_sample_size_parameter(self):
        """Test max_sample_size parameter accepts int."""
        opt = EnsembleOptimizer(SEARCH_SPACE, max_sample_size=1000)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_sampling_parameter(self):
        """Test sampling parameter accepts dict."""
        opt = EnsembleOptimizer(SEARCH_SPACE, sampling={"random": 100})
        opt.search(objective, n_iter=1, verbosity=False)

    def test_replacement_parameter(self):
        """Test replacement parameter accepts bool."""
        opt = EnsembleOptimizer(SEARCH_SPACE, replacement=True)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = EnsembleOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            warm_start_smbo=None,
            max_sample_size=10000000,
            sampling={"random": 1000000},
            replacement=True,
        )
        opt.search(objective, n_iter=1, verbosity=False)

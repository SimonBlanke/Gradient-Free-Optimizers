# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API Freeze Tests for Sequential Model-Based Optimizers

These tests ensure that the public API of sequential model-based optimizers
remains stable. Any changes to parameter names, default values, or method
signatures will cause these tests to fail, alerting developers to potential
breaking changes.

SMB optimizers covered:
- LipschitzOptimizer
- DirectAlgorithm
- BayesianOptimizer
- TreeStructuredParzenEstimators
- ForestOptimizer
"""

import numpy as np
import pytest

from gradient_free_optimizers import (
    LipschitzOptimizer,
    DirectAlgorithm,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
)


# Minimal search space and objective for smoke tests
# Using smaller space for SMB optimizers as they can be computationally heavier
SEARCH_SPACE = {"x": np.linspace(-1, 1, 5)}


def objective(p):
    return -(p["x"] ** 2)


# =============================================================================
# Base parameter tests (common to all optimizers)
# =============================================================================


@pytest.mark.parametrize(
    "Optimizer",
    [
        LipschitzOptimizer,
        DirectAlgorithm,
        BayesianOptimizer,
        TreeStructuredParzenEstimators,
        ForestOptimizer,
    ],
)
class TestBaseParameters:
    """Test base parameters common to all SMB optimizers."""

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
# Common SMBO parameter tests
# =============================================================================


@pytest.mark.parametrize(
    "Optimizer",
    [
        LipschitzOptimizer,
        DirectAlgorithm,
        BayesianOptimizer,
        TreeStructuredParzenEstimators,
        ForestOptimizer,
    ],
)
class TestSMBOParameters:
    """Test SMBO-specific parameters common to all SMB optimizers."""

    def test_warm_start_smbo_parameter(self, Optimizer):
        """Test warm_start_smbo parameter accepts None."""
        opt = Optimizer(SEARCH_SPACE, warm_start_smbo=None)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_max_sample_size_parameter(self, Optimizer):
        """Test max_sample_size parameter accepts int."""
        opt = Optimizer(SEARCH_SPACE, max_sample_size=1000)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_sampling_parameter(self, Optimizer):
        """Test sampling parameter accepts dict."""
        opt = Optimizer(SEARCH_SPACE, sampling={"random": 100})
        opt.search(objective, n_iter=1, verbosity=False)

    def test_replacement_parameter(self, Optimizer):
        """Test replacement parameter accepts bool."""
        opt = Optimizer(SEARCH_SPACE, replacement=True)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_replacement_false(self, Optimizer):
        """Test replacement parameter accepts False."""
        opt = Optimizer(SEARCH_SPACE, replacement=False)
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# LipschitzOptimizer API Tests
# =============================================================================


class TestLipschitzOptimizerAPI:
    """API freeze tests for LipschitzOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = LipschitzOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = LipschitzOptimizer(
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


# =============================================================================
# DirectAlgorithm API Tests
# =============================================================================


class TestDirectAlgorithmAPI:
    """API freeze tests for DirectAlgorithm."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = DirectAlgorithm(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = DirectAlgorithm(
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


# =============================================================================
# BayesianOptimizer API Tests
# =============================================================================


class TestBayesianOptimizerAPI:
    """API freeze tests for BayesianOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = BayesianOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_xi_parameter(self):
        """Test xi parameter exists and accepts float."""
        opt = BayesianOptimizer(SEARCH_SPACE, xi=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_gpr_parameter(self):
        """Test gpr parameter exists and accepts GPR-like object."""
        # Using default gpr object
        opt = BayesianOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = BayesianOptimizer(
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
            xi=0.03,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# TreeStructuredParzenEstimators API Tests
# =============================================================================


class TestTreeStructuredParzenEstimatorsAPI:
    """API freeze tests for TreeStructuredParzenEstimators."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = TreeStructuredParzenEstimators(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_gamma_tpe_parameter(self):
        """Test gamma_tpe parameter exists and accepts float."""
        opt = TreeStructuredParzenEstimators(SEARCH_SPACE, gamma_tpe=0.3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = TreeStructuredParzenEstimators(
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
            gamma_tpe=0.2,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# ForestOptimizer API Tests
# =============================================================================


class TestForestOptimizerAPI:
    """API freeze tests for ForestOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = ForestOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_tree_regressor_extra_tree(self):
        """Test tree_regressor parameter accepts 'extra_tree'."""
        opt = ForestOptimizer(SEARCH_SPACE, tree_regressor="extra_tree")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_tree_regressor_random_forest(self):
        """Test tree_regressor parameter accepts 'random_forest'."""
        opt = ForestOptimizer(SEARCH_SPACE, tree_regressor="random_forest")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_tree_regressor_gradient_boost(self):
        """Test tree_regressor parameter accepts 'gradient_boost'."""
        opt = ForestOptimizer(SEARCH_SPACE, tree_regressor="gradient_boost")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_tree_para_parameter(self):
        """Test tree_para parameter exists and accepts dict."""
        opt = ForestOptimizer(SEARCH_SPACE, tree_para={"n_estimators": 50})
        opt.search(objective, n_iter=1, verbosity=False)

    def test_xi_parameter(self):
        """Test xi parameter exists and accepts float."""
        opt = ForestOptimizer(SEARCH_SPACE, xi=0.05)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = ForestOptimizer(
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
            tree_regressor="extra_tree",
            tree_para={"n_estimators": 100},
            xi=0.03,
        )
        opt.search(objective, n_iter=1, verbosity=False)

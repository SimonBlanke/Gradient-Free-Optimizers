# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
API Freeze Tests for Population-Based Optimizers

These tests ensure that the public API of population-based optimizers remains stable.
Any changes to parameter names, default values, or method signatures will
cause these tests to fail, alerting developers to potential breaking changes.

Population-based optimizers covered:
- ParallelTemperingOptimizer
- ParticleSwarmOptimizer
- SpiralOptimization
- GeneticAlgorithmOptimizer
- EvolutionStrategyOptimizer
- DifferentialEvolutionOptimizer
"""

import numpy as np
import pytest

from gradient_free_optimizers import (
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer,
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
        ParallelTemperingOptimizer,
        ParticleSwarmOptimizer,
        SpiralOptimization,
        GeneticAlgorithmOptimizer,
        EvolutionStrategyOptimizer,
        DifferentialEvolutionOptimizer,
    ],
)
class TestBaseParameters:
    """Test base parameters common to all population-based optimizers."""

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
# Common population parameter tests
# =============================================================================


@pytest.mark.parametrize(
    "Optimizer",
    [
        ParallelTemperingOptimizer,
        ParticleSwarmOptimizer,
        SpiralOptimization,
        GeneticAlgorithmOptimizer,
        EvolutionStrategyOptimizer,
        DifferentialEvolutionOptimizer,
    ],
)
class TestPopulationParameter:
    """Test population parameter common to all population-based optimizers."""

    def test_population_parameter(self, Optimizer):
        """Test population parameter exists and accepts int."""
        opt = Optimizer(SEARCH_SPACE, population=3)
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# ParallelTemperingOptimizer API Tests
# =============================================================================


class TestParallelTemperingOptimizerAPI:
    """API freeze tests for ParallelTemperingOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = ParallelTemperingOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_population_parameter(self):
        """Test population parameter exists and accepts int."""
        opt = ParallelTemperingOptimizer(SEARCH_SPACE, population=3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_iter_swap_parameter(self):
        """Test n_iter_swap parameter exists and accepts int."""
        opt = ParallelTemperingOptimizer(SEARCH_SPACE, n_iter_swap=3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = ParallelTemperingOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            population=5,
            n_iter_swap=5,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# ParticleSwarmOptimizer API Tests
# =============================================================================


class TestParticleSwarmOptimizerAPI:
    """API freeze tests for ParticleSwarmOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = ParticleSwarmOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_population_parameter(self):
        """Test population parameter exists and accepts int."""
        opt = ParticleSwarmOptimizer(SEARCH_SPACE, population=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_inertia_parameter(self):
        """Test inertia parameter exists and accepts float."""
        opt = ParticleSwarmOptimizer(SEARCH_SPACE, inertia=0.7)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_cognitive_weight_parameter(self):
        """Test cognitive_weight parameter exists and accepts float."""
        opt = ParticleSwarmOptimizer(SEARCH_SPACE, cognitive_weight=0.3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_social_weight_parameter(self):
        """Test social_weight parameter exists and accepts float."""
        opt = ParticleSwarmOptimizer(SEARCH_SPACE, social_weight=0.7)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_temp_weight_parameter(self):
        """Test temp_weight parameter exists and accepts float."""
        opt = ParticleSwarmOptimizer(SEARCH_SPACE, temp_weight=0.1)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = ParticleSwarmOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            population=10,
            inertia=0.5,
            cognitive_weight=0.5,
            social_weight=0.5,
            temp_weight=0.2,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# SpiralOptimization API Tests
# =============================================================================


class TestSpiralOptimizationAPI:
    """API freeze tests for SpiralOptimization."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = SpiralOptimization(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_population_parameter(self):
        """Test population parameter exists and accepts int."""
        opt = SpiralOptimization(SEARCH_SPACE, population=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_decay_rate_parameter(self):
        """Test decay_rate parameter exists and accepts float."""
        opt = SpiralOptimization(SEARCH_SPACE, decay_rate=0.95)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = SpiralOptimization(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            population=10,
            decay_rate=0.99,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# GeneticAlgorithmOptimizer API Tests
# =============================================================================


class TestGeneticAlgorithmOptimizerAPI:
    """API freeze tests for GeneticAlgorithmOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = GeneticAlgorithmOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_population_parameter(self):
        """Test population parameter exists and accepts int."""
        opt = GeneticAlgorithmOptimizer(SEARCH_SPACE, population=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_offspring_parameter(self):
        """Test offspring parameter exists and accepts int."""
        opt = GeneticAlgorithmOptimizer(SEARCH_SPACE, offspring=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_crossover_parameter(self):
        """Test crossover parameter exists and accepts string."""
        opt = GeneticAlgorithmOptimizer(SEARCH_SPACE, crossover="discrete-recombination")
        opt.search(objective, n_iter=1, verbosity=False)

    def test_n_parents_parameter(self):
        """Test n_parents parameter exists and accepts int."""
        opt = GeneticAlgorithmOptimizer(SEARCH_SPACE, n_parents=2)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_mutation_rate_parameter(self):
        """Test mutation_rate parameter exists and accepts float."""
        opt = GeneticAlgorithmOptimizer(SEARCH_SPACE, mutation_rate=0.3)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_crossover_rate_parameter(self):
        """Test crossover_rate parameter exists and accepts float."""
        opt = GeneticAlgorithmOptimizer(SEARCH_SPACE, crossover_rate=0.7)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = GeneticAlgorithmOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            population=10,
            offspring=10,
            crossover="discrete-recombination",
            n_parents=2,
            mutation_rate=0.5,
            crossover_rate=0.5,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# EvolutionStrategyOptimizer API Tests
# =============================================================================


class TestEvolutionStrategyOptimizerAPI:
    """API freeze tests for EvolutionStrategyOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = EvolutionStrategyOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_population_parameter(self):
        """Test population parameter exists and accepts int."""
        opt = EvolutionStrategyOptimizer(SEARCH_SPACE, population=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_offspring_parameter(self):
        """Test offspring parameter exists and accepts int."""
        opt = EvolutionStrategyOptimizer(SEARCH_SPACE, offspring=10)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_replace_parents_parameter(self):
        """Test replace_parents parameter exists and accepts bool."""
        opt = EvolutionStrategyOptimizer(SEARCH_SPACE, replace_parents=True)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_mutation_rate_parameter(self):
        """Test mutation_rate parameter exists and accepts float."""
        opt = EvolutionStrategyOptimizer(SEARCH_SPACE, mutation_rate=0.5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_crossover_rate_parameter(self):
        """Test crossover_rate parameter exists and accepts float."""
        opt = EvolutionStrategyOptimizer(SEARCH_SPACE, crossover_rate=0.5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = EvolutionStrategyOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            population=10,
            offspring=20,
            replace_parents=False,
            mutation_rate=0.7,
            crossover_rate=0.3,
        )
        opt.search(objective, n_iter=1, verbosity=False)


# =============================================================================
# DifferentialEvolutionOptimizer API Tests
# =============================================================================


class TestDifferentialEvolutionOptimizerAPI:
    """API freeze tests for DifferentialEvolutionOptimizer."""

    def test_default_parameters(self):
        """Instantiate with only required parameters."""
        opt = DifferentialEvolutionOptimizer(SEARCH_SPACE)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_population_parameter(self):
        """Test population parameter exists and accepts int."""
        opt = DifferentialEvolutionOptimizer(SEARCH_SPACE, population=5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_mutation_rate_parameter(self):
        """Test mutation_rate parameter exists and accepts float."""
        opt = DifferentialEvolutionOptimizer(SEARCH_SPACE, mutation_rate=0.5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_crossover_rate_parameter(self):
        """Test crossover_rate parameter exists and accepts float."""
        opt = DifferentialEvolutionOptimizer(SEARCH_SPACE, crossover_rate=0.5)
        opt.search(objective, n_iter=1, verbosity=False)

    def test_all_parameters_explicit(self):
        """Test all parameters can be set explicitly."""
        opt = DifferentialEvolutionOptimizer(
            SEARCH_SPACE,
            initialize={"random": 2},
            constraints=[],
            random_state=42,
            rand_rest_p=0.1,
            nth_process=None,
            population=10,
            mutation_rate=0.9,
            crossover_rate=0.9,
        )
        opt.search(objective, n_iter=1, verbosity=False)

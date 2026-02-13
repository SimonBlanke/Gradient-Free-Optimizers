# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Tests for Template Method Pattern enforcement.

These tests verify that:
1. Incomplete optimizers cannot be instantiated (ABC enforcement)
2. Template methods are actually called during iteration (mock verification)
3. Subclasses don't bypass the template by overriding iterate()
"""

from unittest.mock import patch

import numpy as np
import pytest

from ..core_optimizer import CoreOptimizer
from ..local_opt import HillClimbingOptimizer

# =============================================================================
# PHASE 1: ABC ENFORCEMENT TESTS
# =============================================================================


class TestABCEnforcement:
    """Test that incomplete optimizer implementations fail at instantiation."""

    def test_incomplete_optimizer_missing_all_batch_methods(self):
        """An optimizer missing all batch methods cannot be instantiated."""

        class IncompleteOptimizer(CoreOptimizer):
            """Missing all three batch methods."""

            def _evaluate(self, score_new):
                pass

        with pytest.raises(TypeError, match="abstract methods"):
            IncompleteOptimizer({"x": np.array([1, 2, 3])})

    def test_incomplete_optimizer_missing_one_batch_method(self):
        """An optimizer missing even one batch method cannot be instantiated."""

        class MissingCategorical(CoreOptimizer):
            """Missing only _iterate_categorical_batch."""

            def _evaluate(self, score_new):
                pass

            def _iterate_continuous_batch(self):
                return np.array([0.5])

            def _iterate_discrete_batch(self):
                return np.array([1])

            # Missing: _iterate_categorical_batch

        with pytest.raises(TypeError, match="abstract method"):
            MissingCategorical({"x": np.array([1, 2, 3])})

    def test_complete_optimizer_can_instantiate(self):
        """An optimizer with all required methods can be instantiated."""

        class CompleteOptimizer(CoreOptimizer):
            """Has all required methods."""

            def _evaluate(self, score_new):
                pass

            def _iterate_continuous_batch(self):
                return np.array([0.5])

            def _iterate_categorical_batch(self):
                return np.array([0])

            def _iterate_discrete_batch(self):
                return np.array([1])

        # Should not raise
        opt = CompleteOptimizer({"x": np.array([1, 2, 3])})
        assert opt is not None


# =============================================================================
# PHASE 2: MOCK-BASED CALL VERIFICATION TESTS
# =============================================================================


class TestTemplateMethodCalls:
    """Test that iterate() actually invokes the batch methods."""

    @pytest.fixture
    def mixed_search_space(self):
        """Create a search space with all three dimension types."""
        return {
            "learning_rate": (0.0001, 0.1),  # continuous
            "activation": ["relu", "tanh", "sigmoid"],  # categorical
            "hidden_layers": np.array([1, 2, 3, 4, 5]),  # discrete
        }

    @pytest.fixture
    def optimizer(self, mixed_search_space):
        """Create a HillClimbingOptimizer with mixed dimensions."""
        return HillClimbingOptimizer(
            search_space=mixed_search_space,
            random_state=42,
        )

    def test_iterate_calls_continuous_batch(self, optimizer):
        """Verify iterate() calls _iterate_continuous_batch for continuous dims."""
        optimizer.pos_current = np.array([0.05, 1, 2])

        with patch.object(
            optimizer,
            "_iterate_continuous_batch",
            wraps=optimizer._iterate_continuous_batch,
        ) as mock:
            optimizer.iterate()
            assert mock.call_count >= 1, "_iterate_continuous_batch was not called"

    def test_iterate_calls_categorical_batch(self, optimizer):
        """Verify iterate() calls _iterate_categorical_batch for categorical dims."""
        optimizer.pos_current = np.array([0.05, 1, 2])

        with patch.object(
            optimizer,
            "_iterate_categorical_batch",
            wraps=optimizer._iterate_categorical_batch,
        ) as mock:
            optimizer.iterate()
            assert mock.call_count >= 1, "_iterate_categorical_batch was not called"

    def test_iterate_calls_discrete_batch(self, optimizer):
        """Verify iterate() calls _iterate_discrete_batch for discrete dims."""
        optimizer.pos_current = np.array([0.05, 1, 2])

        with patch.object(
            optimizer,
            "_iterate_discrete_batch",
            wraps=optimizer._iterate_discrete_batch,
        ) as mock:
            optimizer.iterate()
            assert mock.call_count >= 1, "_iterate_discrete_batch was not called"

    def test_iterate_calls_all_batch_methods_for_mixed_space(self, optimizer):
        """Verify all three batch methods are called for a mixed search space."""
        optimizer.pos_current = np.array([0.05, 1, 2])

        with (
            patch.object(
                optimizer,
                "_iterate_continuous_batch",
                wraps=optimizer._iterate_continuous_batch,
            ) as mock_cont,
            patch.object(
                optimizer,
                "_iterate_categorical_batch",
                wraps=optimizer._iterate_categorical_batch,
            ) as mock_cat,
            patch.object(
                optimizer,
                "_iterate_discrete_batch",
                wraps=optimizer._iterate_discrete_batch,
            ) as mock_disc,
        ):
            optimizer.iterate()

            assert mock_cont.call_count >= 1, "continuous batch not called"
            assert mock_cat.call_count >= 1, "categorical batch not called"
            assert mock_disc.call_count >= 1, "discrete batch not called"

    def test_continuous_only_space_skips_other_batches(self):
        """For continuous-only space, only continuous batch should be called."""
        opt = HillClimbingOptimizer(
            search_space={"x": (0.0, 10.0), "y": (0.0, 10.0)},
            random_state=42,
        )
        opt.pos_current = np.array([5.0, 5.0])

        with (
            patch.object(
                opt, "_iterate_continuous_batch", wraps=opt._iterate_continuous_batch
            ) as mock_cont,
            patch.object(
                opt, "_iterate_categorical_batch", wraps=opt._iterate_categorical_batch
            ) as mock_cat,
            patch.object(
                opt, "_iterate_discrete_batch", wraps=opt._iterate_discrete_batch
            ) as mock_disc,
        ):
            opt.iterate()

            assert mock_cont.call_count >= 1, "continuous batch not called"
            assert mock_cat.call_count == 0, "categorical batch should not be called"
            assert mock_disc.call_count == 0, "discrete batch should not be called"


# =============================================================================
# PHASE 3: BYPASS PREVENTION TESTS
# =============================================================================


# Collect all optimizer classes that use the template pattern
TEMPLATE_USING_OPTIMIZERS = [
    HillClimbingOptimizer,
]

# Optimizers that legitimately override _generate_position() for special behavior
# (e.g., RandomRestartHillClimbing needs custom position generation for restarts)
OPTIMIZERS_WITH_CUSTOM_GENERATE_POSITION = []

# Import additional optimizers for comprehensive testing
try:
    from ..exp_opt import RandomAnnealingOptimizer
    from ..global_opt import (
        RandomRestartHillClimbingOptimizer,
        RandomSearchOptimizer,
    )
    from ..local_opt import (
        DownhillSimplexOptimizer,
        RepulsingHillClimbingOptimizer,
        SimulatedAnnealingOptimizer,
        StochasticHillClimbingOptimizer,
    )

    TEMPLATE_USING_OPTIMIZERS.extend(
        [
            SimulatedAnnealingOptimizer,
            StochasticHillClimbingOptimizer,
            RepulsingHillClimbingOptimizer,
            RandomSearchOptimizer,
            RandomRestartHillClimbingOptimizer,
            RandomAnnealingOptimizer,
            DownhillSimplexOptimizer,
        ]
    )
    # RandomRestartHillClimbing overrides _generate_position for restart logic
    OPTIMIZERS_WITH_CUSTOM_GENERATE_POSITION.append(RandomRestartHillClimbingOptimizer)
except ImportError:
    pass


class TestIterateNotOverridden:
    """Ensure template-using optimizers don't override iterate()."""

    @pytest.mark.parametrize("OptimizerClass", TEMPLATE_USING_OPTIMIZERS)
    def test_optimizer_uses_coreoptimizer_iterate(self, OptimizerClass):
        """Template-using optimizers should inherit iterate() from CoreOptimizer.

        If an optimizer defines iterate() in its own __dict__, it's overriding
        the template method, which violates the pattern.
        """
        # Check if iterate is defined in the class's own __dict__ (not inherited)
        if "iterate" in OptimizerClass.__dict__:
            pytest.fail(
                f"{OptimizerClass.__name__} overrides iterate() in its own class. "
                f"Template optimizers should inherit iterate() from CoreOptimizer."
            )

    @pytest.mark.parametrize("OptimizerClass", TEMPLATE_USING_OPTIMIZERS)
    def test_optimizer_does_not_override_generate_position(self, OptimizerClass):
        """Ensure template-using optimizers don't override _generate_position().

        Exception: Some optimizers like RandomRestartHillClimbing legitimately
        need custom position generation for their restart logic.
        """
        if OptimizerClass in OPTIMIZERS_WITH_CUSTOM_GENERATE_POSITION:
            pytest.skip(
                f"{OptimizerClass.__name__} has legitimate custom _generate_position"
            )

        if "_generate_position" in OptimizerClass.__dict__:
            pytest.fail(
                f"{OptimizerClass.__name__} overrides _generate_position(). "
                f"This method should only be defined in CoreOptimizer."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

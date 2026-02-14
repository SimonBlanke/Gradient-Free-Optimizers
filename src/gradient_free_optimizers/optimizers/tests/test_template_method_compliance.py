"""Tests for Template Method Pattern compliance.

These tests verify that optimizer classes follow the fundamental architecture rules:
- Lifecycle methods must NOT be overridden by concrete optimizers
- Only hook methods (_on_evaluate, _iterate_*_batch, etc.) should be implemented

See DESIGN_EXTENDED_SEARCH_SPACE.md section 4.2.4 for the rules.
"""

import pytest

from ..core_optimizer import CoreOptimizer
from ..exp_opt import (
    RandomAnnealingOptimizer,
)
from ..global_opt import (
    DirectAlgorithm,
    LipschitzOptimizer,
    PatternSearch,
    PowellsMethod,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
)
from ..grid import (
    GridSearchOptimizer,
)

# Local imports from within the optimizers module
from ..local_opt import (
    DownhillSimplexOptimizer,
    HillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticHillClimbingOptimizer,
)
from ..pop_opt import (
    DifferentialEvolutionOptimizer,
    EvolutionStrategyOptimizer,
    GeneticAlgorithmOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
)
from ..smb_opt import (
    BayesianOptimizer,
    ForestOptimizer,
    TreeStructuredParzenEstimators,
)

# =============================================================================
# All backend optimizer classes to test
# =============================================================================

ALL_OPTIMIZERS = [
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    GridSearchOptimizer,
    PatternSearch,
    DirectAlgorithm,
    LipschitzOptimizer,
    PowellsMethod,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    SpiralOptimization,
    GeneticAlgorithmOptimizer,
    EvolutionStrategyOptimizer,
    DifferentialEvolutionOptimizer,
    BayesianOptimizer,
    ForestOptimizer,
    TreeStructuredParzenEstimators,
]


# =============================================================================
# Lifecycle methods that must NOT be overridden
# =============================================================================

# These are the lifecycle methods defined in CoreOptimizer that orchestrate
# the optimization process. They have a FIXED flow and must not be changed.
#
# Per DESIGN_EXTENDED_SEARCH_SPACE.md section 4.2.4, ALL lifecycle methods
# must NOT be overridden. Algorithms needing setup logic should override
# the _on_finish_initialization() HOOK instead.
PUBLIC_METHODS_MUST_NOT_OVERRIDE = [
    "_iterate",
    "_evaluate",
    "_init_pos",
    "_evaluate_init",
    "_finish_initialization",
]

# Private hook methods that CAN be overridden to set up algorithm state
# These are called by the public methods at appropriate times
PRIVATE_HOOKS_MAY_OVERRIDE = [
    "_on_finish_initialization",  # Called by _finish_initialization()
]


def _method_is_defined_in_class(cls, method_name: str) -> bool:
    """Check if a method is defined directly in cls (not just inherited).

    Returns True if the method is defined in cls.__dict__, meaning it was
    written in that class and not inherited from a parent.
    """
    return method_name in cls.__dict__


def _get_method_defining_class(cls, method_name: str):
    """Find which class in the MRO defines the method.

    Returns the class where the method is actually defined.
    """
    for klass in cls.__mro__:
        if method_name in klass.__dict__:
            return klass
    return None


# =============================================================================
# Tests
# =============================================================================


def optimizer_id(opt_class):
    """Generate readable test ID from optimizer class."""
    return opt_class.__name__


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
@pytest.mark.parametrize("method_name", PUBLIC_METHODS_MUST_NOT_OVERRIDE)
def test_lifecycle_method_not_overridden(optimizer_class, method_name):
    """Test that optimizer classes do not override lifecycle methods.

    The Template Method Pattern requires that lifecycle methods (_iterate,
    _evaluate, etc.) are defined in the base class (CoreOptimizer) and have
    a FIXED flow. Concrete optimizer classes must NOT override these methods.

    Instead, they should implement the hook methods:
    - _iterate_continuous_batch()
    - _iterate_categorical_batch()
    - _iterate_discrete_batch()
    - _on_evaluate()

    If this test fails, the optimizer is violating the architecture rules.
    See DESIGN_EXTENDED_SEARCH_SPACE.md section 4.2.4 for details.
    """
    # Check if the method is defined directly in this optimizer class
    if _method_is_defined_in_class(optimizer_class, method_name):
        # Find where in the hierarchy it's defined
        defining_class = _get_method_defining_class(optimizer_class, method_name)

        # If defined in the optimizer itself (not a base class), it's a violation
        if defining_class == optimizer_class:
            pytest.fail(
                f"{optimizer_class.__name__} overrides '{method_name}()'. "
                f"This violates the Template Method Pattern. "
                f"Lifecycle methods must NOT be overridden - only hook methods "
                f"(_iterate_*_batch, _on_evaluate) should be implemented. "
                f"See DESIGN_EXTENDED_SEARCH_SPACE.md section 4.2.4."
            )


@pytest.mark.parametrize("optimizer_class", ALL_OPTIMIZERS, ids=optimizer_id)
def test_inherits_from_core_optimizer(optimizer_class):
    """Test that all optimizers inherit from CoreOptimizer.

    This ensures they get the public method implementations with the fixed flow.
    """
    assert issubclass(optimizer_class, CoreOptimizer), (
        f"{optimizer_class.__name__} does not inherit from CoreOptimizer. "
        f"All optimizers must inherit from CoreOptimizer to get the Template Method "
        f"Pattern implementation."
    )


class TestLifecycleMethodsDefinedInCoreOptimizer:
    """Verify that lifecycle methods are actually defined in CoreOptimizer."""

    @pytest.mark.parametrize("method_name", PUBLIC_METHODS_MUST_NOT_OVERRIDE)
    def test_method_exists_in_core_optimizer(self, method_name):
        """Test that each lifecycle method is defined in CoreOptimizer."""
        assert hasattr(CoreOptimizer, method_name), (
            f"CoreOptimizer is missing lifecycle method '{method_name}()'. "
            f"This method should be defined in CoreOptimizer as part of the "
            f"Template Method Pattern."
        )

    @pytest.mark.parametrize("method_name", PUBLIC_METHODS_MUST_NOT_OVERRIDE)
    def test_method_is_callable(self, method_name):
        """Test that each lifecycle method is callable."""
        method = getattr(CoreOptimizer, method_name, None)
        assert callable(method), (
            f"CoreOptimizer.{method_name} is not callable. " f"It should be a method."
        )


class TestTemplateMethodsExist:
    """Verify that hook methods exist for overriding."""

    TEMPLATE_METHODS = [
        "_iterate_continuous_batch",
        "_iterate_categorical_batch",
        "_iterate_discrete_batch",
        "_on_evaluate",
    ]

    @pytest.mark.parametrize("method_name", TEMPLATE_METHODS)
    def test_hook_method_exists_in_core_optimizer(self, method_name):
        """Test that hook methods are defined in CoreOptimizer.

        These are the methods that concrete optimizers SHOULD override.
        """
        assert hasattr(CoreOptimizer, method_name), (
            f"CoreOptimizer is missing hook method '{method_name}()'. "
            f"This method should be defined in CoreOptimizer for concrete "
            f"optimizers to override."
        )

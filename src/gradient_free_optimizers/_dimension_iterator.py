"""Mixin for dimension-type-specific iteration in optimizers.

This module provides the DimensionIteratorMixin class that enables
optimizers to handle different dimension types (discrete, continuous,
categorical) with appropriate iteration strategies.

The mixin provides:
1. A unified interface for type-aware iteration
2. Default implementations that delegate to CoreOptimizer methods
3. Vectorized implementations for large search spaces (1000+ dimensions)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from ._array_backend import array
from ._array_backend import random as np_random

if TYPE_CHECKING:
    from .optimizers.core_optimizer.converter import ArrayLike

# Threshold for automatic vectorization (number of dimensions)
VECTORIZATION_THRESHOLD = 1000


class DimensionIteratorMixin:
    """Mixin for dimension-type-specific iteration.

    This mixin provides a unified interface for optimizers to handle
    different dimension types. It offers default implementations that
    delegate to the type-aware methods in CoreOptimizer, and abstract
    methods for custom or vectorized implementations.

    Usage
    -----
    Add this mixin to an optimizer class:

        class MyOptimizer(BaseOptimizer, DimensionIteratorMixin):
            def iterate(self):
                # Use type-aware iteration
                return self.iterate_position_typed(
                    self.pos_current,
                    epsilon=self.epsilon
                )

    For custom behavior, override the _iterate_*_all methods:

        class MyOptimizer(BaseOptimizer, DimensionIteratorMixin):
            def _iterate_discrete_numerical_all(self, values, max_pos, **kw):
                # Custom implementation for discrete dimensions
                ...

    Attributes
    ----------
    conv : Converter
        Inherited from CoreOptimizer, provides dimension type information.

    Notes
    -----
    This mixin assumes the class also inherits from CoreOptimizer or has
    access to self.conv (Converter) and self.move_climb_typed() etc.
    """

    def iterate_position_typed(
        self,
        current_pos: ArrayLike,
        use_vectorized: bool = False,
        **kwargs: Any,
    ) -> ArrayLike:
        """Generate a new position based on dimension types.

        This method orchestrates the iteration process by selecting the
        appropriate strategy based on dimension types and vectorization
        preference.

        Parameters
        ----------
        current_pos : array-like
            The current position to iterate from.
        use_vectorized : bool, default False
            If True and vectorization is available, use vectorized
            implementation for better performance with large search spaces.
        **kwargs : dict
            Algorithm-specific parameters (epsilon, distribution, etc.)

        Returns
        -------
        array-like
            New position satisfying all constraints.
        """
        if use_vectorized and self._can_vectorize():
            return self._iterate_position_vectorized(current_pos, **kwargs)
        else:
            return self._iterate_position_sequential(current_pos, **kwargs)

    def _iterate_position_sequential(
        self,
        current_pos: ArrayLike,
        **kwargs: Any,
    ) -> ArrayLike:
        """Sequential iteration using CoreOptimizer's type-aware methods.

        This is the default implementation that delegates to move_climb_typed
        from CoreOptimizer. Subclasses can override for custom behavior.
        """
        epsilon = kwargs.get("epsilon", getattr(self, "epsilon", 0.03))
        distribution = kwargs.get(
            "distribution", getattr(self, "distribution", "normal")
        )

        # Delegate to CoreOptimizer's type-aware method
        return self.move_climb_typed(current_pos, epsilon, distribution)

    def _can_vectorize(self) -> bool:
        """Check if vectorized iteration is available and beneficial.

        Returns True if:
        1. The search space has more dimensions than VECTORIZATION_THRESHOLD
        2. The mixin has vectorized implementations available

        Override this method to customize vectorization behavior.

        Returns
        -------
        bool
            True if vectorized iteration should be used.
        """
        # Check if we have enough dimensions to benefit from vectorization
        if not hasattr(self, "conv"):
            return False
        return self.conv.n_dimensions >= VECTORIZATION_THRESHOLD

    def _iterate_position_vectorized(
        self,
        current_pos: ArrayLike,
        **kwargs: Any,
    ) -> ArrayLike:
        """Vectorized iteration over all dimensions.

        This method processes all dimensions of the same type in a single
        array operation, which is much faster for large search spaces
        (e.g., 100M+ dimensions).

        Override the _iterate_*_all methods to provide vectorized
        implementations for each dimension type.

        Parameters
        ----------
        current_pos : array-like
            Current position.
        **kwargs : dict
            Algorithm-specific parameters.

        Returns
        -------
        array-like
            New position.
        """
        from ._array_backend import array

        new_pos = list(current_pos)
        masks = self.conv.dim_masks

        # Process all discrete-numerical dimensions together
        if masks.has_discrete_numerical:
            discrete_values = [current_pos[i] for i in masks.discrete_numerical]
            max_positions = [
                self.conv.dim_infos[i].bounds[1] for i in masks.discrete_numerical
            ]
            new_discrete = self._iterate_discrete_numerical_all(
                discrete_values, max_positions, **kwargs
            )
            for idx, dim_idx in enumerate(masks.discrete_numerical):
                new_pos[dim_idx] = new_discrete[idx]

        # Process all continuous dimensions together
        if masks.has_continuous:
            continuous_values = [current_pos[i] for i in masks.continuous]
            bounds = [self.conv.dim_infos[i].bounds for i in masks.continuous]
            new_continuous = self._iterate_continuous_all(
                continuous_values, bounds, **kwargs
            )
            for idx, dim_idx in enumerate(masks.continuous):
                new_pos[dim_idx] = new_continuous[idx]

        # Process all categorical dimensions together
        if masks.has_categorical:
            categorical_values = [current_pos[i] for i in masks.categorical]
            n_categories = [self.conv.dim_infos[i].size for i in masks.categorical]
            new_categorical = self._iterate_categorical_all(
                categorical_values, n_categories, **kwargs
            )
            for idx, dim_idx in enumerate(masks.categorical):
                new_pos[dim_idx] = new_categorical[idx]

        return self.conv2pos_typed(array(new_pos))

    def _iterate_discrete_numerical_all(
        self,
        current_values: list[float],
        max_positions: list[int],
        **kwargs: Any,
    ) -> list[float]:
        """Iterate all discrete-numerical dimensions together (vectorized).

        Applies Gaussian noise scaled by dimension size to all discrete
        dimensions in a single array operation.

        Parameters
        ----------
        current_values : list
            Current values for all discrete-numerical dimensions.
        max_positions : list
            Maximum position index for each dimension.
        **kwargs : dict
            Algorithm-specific parameters (epsilon, epsilon_mod).

        Returns
        -------
        list
            New values for all discrete-numerical dimensions.
        """
        epsilon = kwargs.get("epsilon", getattr(self, "epsilon", 0.03))
        epsilon_mod = kwargs.get("epsilon_mod", 1.0)

        # Convert to arrays for vectorized operations
        values = array(current_values)
        max_pos = array(max_positions)

        # Generate Gaussian noise scaled by dimension sizes
        sigmas = max_pos * epsilon * epsilon_mod
        # Use array of random values
        noise = array([np_random.normal(0, float(s)) for s in sigmas])

        new_values = values + noise
        return list(new_values)

    def _iterate_continuous_all(
        self,
        current_values: list[float],
        bounds: list[tuple[float, float]],
        **kwargs: Any,
    ) -> list[float]:
        """Iterate all continuous dimensions together (vectorized).

        Applies Gaussian noise scaled by range to all continuous
        dimensions in a single array operation.

        Parameters
        ----------
        current_values : list
            Current values for all continuous dimensions.
        bounds : list
            List of (min, max) tuples for each dimension.
        **kwargs : dict
            Algorithm-specific parameters (epsilon, epsilon_mod).

        Returns
        -------
        list
            New values for all continuous dimensions.
        """
        epsilon = kwargs.get("epsilon", getattr(self, "epsilon", 0.03))
        epsilon_mod = kwargs.get("epsilon_mod", 1.0)

        # Convert to arrays for vectorized operations
        values = array(current_values)
        ranges = array([b[1] - b[0] for b in bounds])

        # Generate Gaussian noise scaled by ranges
        sigmas = ranges * epsilon * epsilon_mod
        noise = array([np_random.normal(0, float(s)) for s in sigmas])

        new_values = values + noise
        return list(new_values)

    def _iterate_categorical_all(
        self,
        current_values: list[int],
        n_categories: list[int],
        **kwargs: Any,
    ) -> list[int]:
        """Iterate all categorical dimensions together (vectorized).

        Applies probabilistic category switching to all categorical
        dimensions.

        Parameters
        ----------
        current_values : list
            Current category indices for all categorical dimensions.
        n_categories : list
            Number of categories for each dimension.
        **kwargs : dict
            Algorithm-specific parameters (epsilon, epsilon_mod).

        Returns
        -------
        list
            New category indices for all categorical dimensions.
        """
        epsilon = kwargs.get("epsilon", getattr(self, "epsilon", 0.03))
        epsilon_mod = kwargs.get("epsilon_mod", 1.0)
        switch_prob = epsilon * epsilon_mod

        new_values = []
        for val, n_cat in zip(current_values, n_categories):
            if random.random() < switch_prob:  # noqa: S311
                # Switch to random category
                new_values.append(random.randint(0, n_cat - 1))  # noqa: S311
            else:
                # Keep current category
                new_values.append(int(val))

        return new_values

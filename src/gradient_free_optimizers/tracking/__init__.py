"""
Experiment tracking module for Gradient-Free-Optimizers.

This module provides tools for tracking, analyzing, and visualizing
optimization experiments. Data is stored in a SQLite database and
can be analyzed using built-in plotting methods.

Example:
    from gradient_free_optimizers import HillClimbingOptimizer
    from gradient_free_optimizers.tracking import SearchTracker

    # Create tracker (data stored in SQLite)
    tracker = SearchTracker("my_experiment.db")

    # Decorate objective function
    @tracker.track
    def objective(x, y):
        return -(x**2 + y**2)

    # Run optimization as usual
    search_space = {"x": (-5, 5), "y": (-5, 5)}
    opt = HillClimbingOptimizer(search_space)
    opt.search(objective, n_iter=100)

    # Analyze results
    print(tracker.summary())
    print(f"Best: {tracker.best_parameters} -> {tracker.best_score}")

    # Visualize
    tracker.plot.convergence()
    tracker.plot.search_space()
    tracker.plot.parameter_importance()

    # Data persists - load later for further analysis
    tracker = SearchTracker.load("my_experiment.db")
    tracker.plot.parallel_coordinates()
"""

from .record import EvaluationRecord, ExperimentMetadata
from .storage import SQLiteBackend, StorageBackend
from .tracker import SearchTracker

__all__ = [
    "SearchTracker",
    "EvaluationRecord",
    "ExperimentMetadata",
    "StorageBackend",
    "SQLiteBackend",
]

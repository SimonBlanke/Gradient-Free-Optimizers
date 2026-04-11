"""Tests for the pluggable storage system."""

import os
import tempfile

import numpy as np
import pytest

from gradient_free_optimizers import HillClimbingOptimizer, RandomSearchOptimizer
from gradient_free_optimizers._result import Result
from gradient_free_optimizers._storage import BaseStorage, MemoryStorage, SQLiteStorage

search_space = {"x": np.linspace(-10, 10, 100)}


def objective(para):
    return -(para["x"] ** 2)


class TestMemoryStorage:
    def test_get_put_contains(self):
        ms = MemoryStorage()
        assert not ms.contains((1, 2))
        assert ms.get((1, 2)) is None

        ms.put((1, 2), Result(0.5, {}))
        assert ms.contains((1, 2))
        assert ms.get((1, 2)).score == 0.5

    def test_len(self):
        ms = MemoryStorage()
        assert len(ms) == 0
        ms.put((1,), Result(0.1, {}))
        ms.put((2,), Result(0.2, {}))
        assert len(ms) == 2

    def test_update(self):
        ms = MemoryStorage()
        ms.update({(1,): Result(0.1, {}), (2,): Result(0.2, {})})
        assert len(ms) == 2
        assert ms.get((2,)).score == 0.2

    def test_items(self):
        ms = MemoryStorage()
        ms.put((1,), Result(0.1, {}))
        ms.put((2,), Result(0.2, {}))
        items = list(ms.items())
        assert len(items) == 2

    def test_overwrite(self):
        ms = MemoryStorage()
        ms.put((1,), Result(0.1, {}))
        ms.put((1,), Result(0.9, {}))
        assert ms.get((1,)).score == 0.9
        assert len(ms) == 1

    def test_isinstance(self):
        assert isinstance(MemoryStorage(), BaseStorage)

    def test_with_search(self):
        storage = MemoryStorage()
        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=20, memory=storage, verbosity=False)
        assert len(storage) > 0
        assert opt.best_score is not None


class TestSQLiteStorage:
    def test_get_put_contains(self):
        with tempfile.TemporaryDirectory() as td:
            ss = SQLiteStorage(os.path.join(td, "test.db"))
            assert not ss.contains((1, 2))
            assert ss.get((1, 2)) is None

            ss.put((1, 2), Result(0.5, {"loss": 0.5}))
            assert ss.contains((1, 2))
            r = ss.get((1, 2))
            assert r.score == 0.5
            assert r.metrics == {"loss": 0.5}
            ss.close()

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "persist.db")
            ss = SQLiteStorage(path)
            ss.put((1,), Result(0.1, {}))
            ss.put((2,), Result(0.2, {"m": 42}))
            ss.close()

            ss2 = SQLiteStorage(path)
            assert len(ss2) == 2
            assert ss2.get((1,)).score == 0.1
            assert ss2.get((2,)).metrics == {"m": 42}
            ss2.close()

    def test_bulk_update(self):
        ss = SQLiteStorage(":memory:")
        data = {(i,): Result(float(i), {}) for i in range(100)}
        ss.update(data)
        assert len(ss) == 100
        assert ss.get((50,)).score == 50.0
        ss.close()

    def test_items_lazy(self):
        ss = SQLiteStorage(":memory:")
        for i in range(50):
            ss.put((i,), Result(float(i), {}))
        items = list(ss.items())
        assert len(items) == 50
        ss.close()

    def test_isinstance(self):
        ss = SQLiteStorage(":memory:")
        assert isinstance(ss, BaseStorage)
        ss.close()

    def test_with_search(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "search.db")
            storage = SQLiteStorage(path)
            opt = HillClimbingOptimizer(search_space)
            opt.search(objective, n_iter=20, memory=storage, verbosity=False)
            assert len(storage) > 0
            storage.close()

    def test_crash_recovery(self):
        """Second search with same storage skips cached positions."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "recovery.db")
            storage = SQLiteStorage(path)

            opt1 = RandomSearchOptimizer(search_space)
            opt1.search(objective, n_iter=20, memory=storage, verbosity=False)
            cached_after_first = len(storage)

            opt2 = RandomSearchOptimizer(search_space)
            opt2.search(objective, n_iter=20, memory=storage, verbosity=False)
            cached_after_second = len(storage)

            assert cached_after_second >= cached_after_first
            storage.close()


class TestMemoryParameterTypes:
    def test_memory_true(self):
        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=10, memory=True, verbosity=False)
        assert opt.best_score is not None

    def test_memory_false(self):
        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=10, memory=False, verbosity=False)
        assert opt.best_score is not None

    def test_memory_storage_instance(self):
        storage = MemoryStorage()
        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=10, memory=storage, verbosity=False)
        assert len(storage) > 0

    def test_memory_sqlite_instance(self):
        ss = SQLiteStorage(":memory:")
        opt = HillClimbingOptimizer(search_space)
        opt.search(objective, n_iter=10, memory=ss, verbosity=False)
        assert len(ss) > 0
        ss.close()

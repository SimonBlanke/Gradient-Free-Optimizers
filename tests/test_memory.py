import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from gradient_free_optimizers import RandomSearchOptimizer
from sklearn.ensemble import GradientBoostingClassifier


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 10, 1),
}


def test_memory_timeSave_0():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(para):
        dtc = DecisionTreeClassifier(min_samples_split=para["min_samples_split"])
        scores = cross_val_score(dtc, X, y, cv=5)

        return scores.mean()

    search_space = {
        "min_samples_split": np.arange(2, 20),
    }

    c_time1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100)
    diff_time1 = time.time() - c_time1

    c_time2 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=100, memory=False)
    diff_time2 = time.time() - c_time2

    print("\n diff_time1 ", diff_time1)
    print("\n diff_time2 ", diff_time2)

    assert diff_time1 < diff_time2 * 0.5


def test_memory_timeSave_1():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(para):
        dtc = DecisionTreeClassifier(max_depth=para["max_depth"])
        scores = cross_val_score(dtc, X, y, cv=10)

        return scores.mean()

    search_space = {
        "max_depth": np.arange(1, 101),
    }

    results = pd.DataFrame(np.arange(1, 101), columns=["max_depth"])
    results["score"] = 0

    c_time1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=300, memory_warm_start=results)
    diff_time1 = time.time() - c_time1
    print("\n diff_time1 ", diff_time1)

    assert diff_time1 < 0.5


def test_memory_warm_start():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(para):
        dtc = DecisionTreeClassifier(
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
        )
        scores = cross_val_score(dtc, X, y, cv=5)

        return scores.mean()

    search_space = {
        "max_depth": np.arange(1, 10),
        "min_samples_split": np.arange(2, 20),
    }

    c_time1 = time.time()
    opt0 = RandomSearchOptimizer(search_space)
    opt0.search(objective_function, n_iter=300)
    diff_time1 = time.time() - c_time1

    c_time2 = time.time()
    opt1 = RandomSearchOptimizer(search_space)
    opt1.search(objective_function, n_iter=300, memory_warm_start=opt0.search_data)
    diff_time2 = time.time() - c_time2

    print("\n diff_time1 ", diff_time1)
    print("\n diff_time2 ", diff_time2)

    assert diff_time2 < diff_time1 * 0.5


def test_memory_warm_start_manual():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(para):
        dtc = GradientBoostingClassifier(
            n_estimators=para["n_estimators"],
        )
        scores = cross_val_score(dtc, X, y, cv=5)

        return scores.mean()

    search_space = {
        "n_estimators": np.arange(500, 502),
    }

    c_time_1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=1)
    diff_time_1 = time.time() - c_time_1

    memory_warm_start = pd.DataFrame(
        [[500, 0.9], [501, 0.91]], columns=["n_estimators", "score"]
    )

    c_time = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=30, memory_warm_start=memory_warm_start)
    diff_time_2 = time.time() - c_time

    print("\n diff_time1 ", diff_time_1)
    print("\n diff_time2 ", diff_time_2)

    assert diff_time_1 * 0.5 > diff_time_2


def test_memory_warm_start_wrong_type():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(para):
        dtc = GradientBoostingClassifier(
            n_estimators=para["n_estimators"],
        )
        scores = cross_val_score(dtc, X, y, cv=3)

        return scores.mean()

    search_space = {
        "n_estimators": np.arange(500, 502),
    }

    memory_warm_start = pd.DataFrame(
        [[500, 0.9], [501, 0.91]], columns=["n_estimators", "score"]
    )

    opt = RandomSearchOptimizer(search_space)
    opt.search(
        objective_function, n_iter=10, memory_warm_start=memory_warm_start.values
    )


def test_memory_warm_start_wrong_search_space():
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective_function(para):
        dtc = GradientBoostingClassifier(
            n_estimators=para["n_estimators"],
        )
        scores = cross_val_score(dtc, X, y, cv=3)

        return scores.mean()

    search_space = {
        "n_estimators": np.arange(400, 402),
    }

    c_time_1 = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=1)
    diff_time_1 = time.time() - c_time_1

    memory_warm_start = pd.DataFrame(
        [[500, 0.9], [501, 0.91]], columns=["n_estimators", "score"]
    )

    c_time = time.time()
    opt = RandomSearchOptimizer(search_space)
    opt.search(objective_function, n_iter=10, memory_warm_start=memory_warm_start)
    diff_time = time.time() - c_time

    assert (diff_time_1 - diff_time) < diff_time / 10

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine

from gradient_free_optimizers import HillClimbingOptimizer


data = load_wine()
X, y = data.data, data.target


def model(para):
    gbc = DecisionTreeClassifier(
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(gbc, X, y, cv=5)

    return scores.mean()


search_space = {
    "min_samples_split": np.arange(2, 25, 1),
    "min_samples_leaf": np.arange(1, 25, 1),
}

opt = HillClimbingOptimizer(search_space)
opt.search(model, n_iter=500, memory=False)


print("\n\nMemory activated:")
opt = HillClimbingOptimizer(search_space)
opt.search(model, n_iter=500, memory=True)

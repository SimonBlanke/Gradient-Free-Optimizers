======================
Hyperparameter Tuning
======================

This example shows how to use GFO for machine learning hyperparameter optimization.


Random Forest Tuning
--------------------

.. code-block:: python

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris
    from gradient_free_optimizers import BayesianOptimizer

    # Load data
    X, y = load_iris(return_X_y=True)

    # Define objective: cross-validation accuracy
    def objective(para):
        clf = RandomForestClassifier(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            min_samples_leaf=para["min_samples_leaf"],
            random_state=42,
        )
        return cross_val_score(clf, X, y, cv=5).mean()

    # Define search space
    search_space = {
        "n_estimators": np.arange(10, 200, 10),
        "max_depth": np.arange(2, 20),
        "min_samples_split": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 10),
    }

    # Run Bayesian optimization
    opt = BayesianOptimizer(search_space, random_state=42)
    opt.search(
        objective,
        n_iter=50,
        verbosity=["progress_bar", "print_results"],
    )

    print(f"\nBest accuracy: {opt.best_score:.4f}")
    print(f"Best parameters: {opt.best_para}")


SVM with Mixed Parameters
-------------------------

.. code-block:: python

    import numpy as np
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from gradient_free_optimizers import TreeStructuredParzenEstimators

    # Load and preprocess data
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    def objective(para):
        clf = SVC(
            C=para["C"],
            kernel=para["kernel"],
            gamma=para["gamma"],
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    # Mixed search space: continuous, discrete, and categorical
    search_space = {
        "C": np.logspace(-2, 2, 50),          # Continuous (log scale)
        "kernel": np.array(["linear", "rbf", "poly"]),  # Categorical
        "gamma": np.logspace(-3, 0, 30),      # Continuous
    }

    # TPE handles mixed spaces well
    opt = TreeStructuredParzenEstimators(search_space)
    opt.search(objective, n_iter=40)

    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Best parameters: {opt.best_para}")


Neural Network Hyperparameters
------------------------------

.. code-block:: python

    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_digits
    from gradient_free_optimizers import ForestOptimizer

    X, y = load_digits(return_X_y=True)

    def objective(para):
        # Build hidden layer sizes from individual parameters
        hidden_layers = (para["layer1"], para["layer2"])

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=para["learning_rate"],
            alpha=para["alpha"],
            max_iter=200,
            random_state=42,
        )
        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "layer1": np.arange(32, 256, 32),
        "layer2": np.arange(16, 128, 16),
        "learning_rate": np.logspace(-4, -1, 20),
        "alpha": np.logspace(-5, -1, 20),
    }

    opt = ForestOptimizer(search_space)
    opt.search(objective, n_iter=30)

    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Best config: {opt.best_para}")


Comparing Multiple Models
-------------------------

.. code-block:: python

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_breast_cancer
    from gradient_free_optimizers import BayesianOptimizer

    X, y = load_breast_cancer(return_X_y=True)

    def objective(para):
        model_type = para["model_type"]

        if model_type == 0:  # Random Forest
            clf = RandomForestClassifier(
                n_estimators=para["n_estimators"],
                max_depth=para["max_depth"],
                random_state=42,
            )
        elif model_type == 1:  # Gradient Boosting
            clf = GradientBoostingClassifier(
                n_estimators=para["n_estimators"],
                max_depth=para["max_depth"],
                learning_rate=para["learning_rate"],
                random_state=42,
            )
        else:  # SVM
            clf = SVC(C=para["C"], kernel="rbf", random_state=42)

        return cross_val_score(clf, X, y, cv=3).mean()

    search_space = {
        "model_type": np.array([0, 1, 2]),  # 0=RF, 1=GB, 2=SVM
        "n_estimators": np.arange(50, 200, 25),
        "max_depth": np.arange(2, 15),
        "learning_rate": np.linspace(0.01, 0.3, 15),
        "C": np.logspace(-1, 2, 20),
    }

    opt = BayesianOptimizer(search_space)
    opt.search(objective, n_iter=50)

    model_names = ["RandomForest", "GradientBoosting", "SVM"]
    print(f"Best model: {model_names[opt.best_para['model_type']]}")
    print(f"Best accuracy: {opt.best_score:.4f}")
    print(f"Best parameters: {opt.best_para}")

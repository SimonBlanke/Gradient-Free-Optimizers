.. raw:: html

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@200;400;600&display=swap" rel="stylesheet">

    <div class="hero-section">
        <h1 class="hero-title" style="font-family: 'Outfit', sans-serif; font-weight: 200;">
            Gradient Free Optimizers
        </h1>
        <p class="hero-tagline">
            Lightweight optimization with local, global, population-based and sequential techniques across mixed search spaces
        </p>
    </div>

    <div class="stats-strip">
        <a href="api_reference/index.html" class="stat-item">
            <div class="stat-front">
                <div class="stat-value">22</div>
                <div class="stat-label">Algorithms</div>
            </div>
            <div class="stat-hover">
                <div class="stat-hover-text">Hill Climbing · Bayesian · Particle Swarm · Genetic · Simulated Annealing ...</div>
            </div>
        </a>
        <a href="api_reference/index.html" class="stat-item">
            <div class="stat-front">
                <div class="stat-value">mixed</div>
                <div class="stat-label">Search Spaces</div>
            </div>
            <div class="stat-hover">
                <div class="stat-hover-text">Continuous, discrete, and categorical dimensions automatically detected</div>
            </div>
        </a>
        <div class="stat-item">
            <div class="stat-front">
                <div class="stat-value">minimal</div>
                <div class="stat-label">Dependencies</div>
            </div>
            <div class="stat-hover">
                <div class="stat-hover-text">Only NumPy and pandas required</div>
            </div>
        </div>
        <div class="stat-item">
            <div class="stat-front">
                <div class="stat-value">pythonic</div>
                <div class="stat-label">API</div>
            </div>
            <div class="stat-hover">
                <div class="stat-hover-text">Dict search spaces, callable objectives, DataFrame results</div>
            </div>
        </div>
    </div>

    <div class="badge-strip">
        <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers/actions">
            <img src="https://github.com/SimonBlanke/Gradient-Free-Optimizers/actions/workflows/tests.yml/badge.svg" alt="Tests">
        </a>
        <a href="https://codecov.io/gh/SimonBlanke/Gradient-Free-Optimizers">
            <img src="https://codecov.io/gh/SimonBlanke/Gradient-Free-Optimizers/branch/master/graph/badge.svg" alt="Coverage">
        </a>
        <a href="https://pypi.org/project/gradient-free-optimizers/">
            <img src="https://img.shields.io/pypi/v/gradient-free-optimizers.svg" alt="PyPI">
        </a>
        <a href="https://pepy.tech/project/gradient-free-optimizers">
            <img src="https://static.pepy.tech/badge/gradient-free-optimizers" alt="Downloads">
        </a>
    </div>


Why Gradient-Free?
------------------

Not every optimization problem has gradients. Machine learning hyperparameter tuning,
simulation optimization, feature selection, and black-box optimization all require
algorithms that can navigate parameter spaces without derivative information.

**Gradient-Free-Optimizers** provides a unified interface to 22 optimization algorithms,
from simple hill climbing to sophisticated Bayesian optimization, all designed for
discrete, continuous, and mixed search spaces.


Algorithm Categories
--------------------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Local Search
        :link: user_guide/optimizers/local/index
        :link-type: doc
        :class-card: sd-border-start sd-border-danger

        **5 Algorithms** for exploiting promising regions

        Hill Climbing, Simulated Annealing, Downhill Simplex, and more.
        These algorithms excel at fine-tuning solutions in the neighborhood
        of good starting points.

    .. grid-item-card:: Global Search
        :link: user_guide/optimizers/global/index
        :link-type: doc
        :class-card: sd-border-start sd-border-success

        **8 Algorithms** for broad exploration

        Random Search, Grid Search, Pattern Search, Powell's Method,
        Lipschitz Optimization, and DIRECT. Designed for thorough coverage
        of the entire search space.

    .. grid-item-card:: Population-Based
        :link: user_guide/optimizers/population/index
        :link-type: doc
        :class-card: sd-border-start sd-border-primary

        **6 Algorithms** using collective intelligence

        Particle Swarm, Genetic Algorithm, Evolution Strategy,
        Differential Evolution, and more. Multiple agents work together
        to find optimal solutions.

    .. grid-item-card:: Sequential Model-Based
        :link: user_guide/optimizers/smbo/index
        :link-type: doc
        :class-card: sd-border-start sd-border-warning

        **3 Algorithms** that learn from evaluations

        Bayesian Optimization, TPE, and Forest Optimizer.
        Build surrogate models to predict promising regions and
        balance exploration vs. exploitation.


Key Features
------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: Simple Search Space Definition
        :class-card: sd-rounded-3

        Define search spaces using NumPy arrays. No complex syntax,
        no special types. Supports discrete, continuous, and mixed spaces.

        .. code-block:: python

            search_space = {
                "x": np.linspace(-10, 10, 100),
                "y": np.arange(1, 100),
                "method": ["adam", "sgd", "rmsprop"],
            }

    .. grid-item-card:: Dual API: search() & ask-tell
        :class-card: sd-rounded-3

        Use the simple ``search()`` method for quick optimization,
        or the ``ask()``/``tell()`` interface for full control over
        the optimization loop.

        .. code-block:: python

            # Simple API
            opt.search(objective, n_iter=100)

            # Ask-tell API
            params = opt.ask()
            score = objective(params)
            opt.tell(params, score)

    .. grid-item-card:: Constraint Support
        :class-card: sd-rounded-3

        Define constraints as simple Python functions. The optimizer
        automatically respects constraints during search, retrying
        invalid positions.

        .. code-block:: python

            constraints = [
                lambda p: p["x"] + p["y"] < 100,
                lambda p: p["x"] > 0,
            ]
            opt = Optimizer(space, constraints=constraints)

    .. grid-item-card:: Memory & Warm Starts
        :class-card: sd-rounded-3

        Cache expensive function evaluations automatically.
        Continue previous searches using ``memory_warm_start``
        with a DataFrame of past evaluations.

        .. code-block:: python

            opt.search(objective, memory=True)

            # Later, continue with warm start
            opt.search(objective,
                      memory_warm_start=previous_data)

    .. grid-item-card:: Multiple Stopping Conditions
        :class-card: sd-rounded-3

        Stop optimization based on iteration count, wall-clock time,
        target score, or early stopping when no improvement is found.
        Combine conditions as needed.

        .. code-block:: python

            opt.search(
                objective,
                n_iter=1000,
                max_time=3600,
                max_score=0.99,
                early_stopping={"n_iter_no_change": 50}
            )

    .. grid-item-card:: Flexible Initialization
        :class-card: sd-rounded-3

        Control how the search starts: grid sampling, random points,
        vertices/corners, or specific warm-start positions you define.

        .. code-block:: python

            opt = Optimizer(
                search_space,
                initialize={
                    "grid": 4,
                    "random": 2,
                    "vertices": 4,
                    "warm_start": [{"x": 0, "y": 50}],
                }
            )


Quick Start
-----------

.. tab-set::

    .. tab-item:: Install

        .. code-block:: bash

            pip install gradient-free-optimizers

    .. tab-item:: Basic Example

        .. code-block:: python

            import numpy as np
            from gradient_free_optimizers import HillClimbingOptimizer

            def objective(para):
                return -(para["x"] ** 2 + para["y"] ** 2)

            search_space = {
                "x": np.linspace(-10, 10, 100),
                "y": np.linspace(-10, 10, 100),
            }

            opt = HillClimbingOptimizer(search_space)
            opt.search(objective, n_iter=1000)

            print(opt.best_para)   # Best parameters found
            print(opt.best_score)  # Best score achieved

    .. tab-item:: Hyperparameter Tuning

        .. code-block:: python

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.datasets import load_iris
            from gradient_free_optimizers import BayesianOptimizer

            X, y = load_iris(return_X_y=True)

            def objective(para):
                clf = RandomForestClassifier(
                    n_estimators=para["n_estimators"],
                    max_depth=para["max_depth"],
                    min_samples_split=para["min_samples_split"],
                )
                return cross_val_score(clf, X, y, cv=5).mean()

            search_space = {
                "n_estimators": np.arange(10, 200, 10),
                "max_depth": np.arange(2, 20),
                "min_samples_split": np.arange(2, 20),
            }

            opt = BayesianOptimizer(search_space)
            opt.search(objective, n_iter=50)

            print(f"Best accuracy: {opt.best_score:.3f}")
            print(f"Best params: {opt.best_para}")

    .. tab-item:: Ask-Tell Interface

        .. code-block:: python

            import numpy as np
            from gradient_free_optimizers import ParticleSwarmOptimizer

            def objective(para):
                return -(para["x"] ** 2 + para["y"] ** 2)

            search_space = {
                "x": np.linspace(-10, 10, 100),
                "y": np.linspace(-10, 10, 100),
            }

            opt = ParticleSwarmOptimizer(search_space, population=10)
            opt.setup_search(objective, n_iter=100)

            for i in range(100):
                params = opt.ask()
                score = objective(params)
                opt.tell(params, score)

                if i % 20 == 0:
                    print(f"Iter {i}: best score = {opt.best_score:.4f}")


The GFO Ecosystem
-----------------

Gradient-Free-Optimizers serves as the optimization backend for
`Hyperactive <https://github.com/SimonBlanke/Hyperactive>`_, a broader
tool for hyperparameter optimization and experiment tracking.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Gradient-Free-Optimizers
        :class-card: sd-rounded-3 sd-border-start sd-border-info

        **You are here**

        Pure optimization algorithms with minimal dependencies.
        Perfect when you need fine-grained control over the
        optimization process.

        - 22 algorithms
        - NumPy-based search spaces
        - Constraint support
        - Ask-tell interface

    .. grid-item-card:: Hyperactive
        :link: https://github.com/SimonBlanke/Hyperactive
        :class-card: sd-rounded-3 sd-border-start sd-border-secondary

        **Higher-level optimization**

        Built on GFO, adding features for ML hyperparameter tuning:
        multi-process optimization, experiment tracking, and
        integration with scikit-learn, PyTorch, and more.

        - Parallel optimization
        - Multiple backends (GFO, Optuna, sklearn)
        - Framework integrations
        - Experiment tracking


Documentation
-------------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: Get Started
        :link: get_started
        :link-type: doc
        :text-align: center

        Quick installation and first optimization

    .. grid-item-card:: User Guide
        :link: user_guide/index
        :link-type: doc
        :text-align: center

        In-depth tutorials and concepts

    .. grid-item-card:: Algorithms
        :link: user_guide/optimizers/index
        :link-type: doc
        :text-align: center

        All 22 optimization algorithms explained

    .. grid-item-card:: Examples
        :link: examples/index
        :link-type: doc
        :text-align: center

        Code examples for common use cases

    .. grid-item-card:: API Reference
        :link: api_reference/index
        :link-type: doc
        :text-align: center

        Complete API documentation

    .. grid-item-card:: FAQ
        :link: faq/index
        :link-type: doc
        :text-align: center

        Frequently asked questions


.. raw:: html

   <script>
   document.addEventListener('DOMContentLoaded', function() {
      const sidebar = document.querySelector('.bd-toc-nav.page-toc');
      if (sidebar) {
         const backToTopDiv = document.createElement('div');
         backToTopDiv.className = 'back-to-top-sidebar';
         backToTopDiv.innerHTML = '<a href="#">\u2191 Back to top</a>';
         sidebar.appendChild(backToTopDiv);

         backToTopDiv.querySelector('a').addEventListener('click', function(e) {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
         });

         window.addEventListener('scroll', function() {
            if (window.scrollY > 400) {
               backToTopDiv.classList.add('visible');
            } else {
               backToTopDiv.classList.remove('visible');
            }
         });
      }
   });
   </script>


.. toctree::
    :maxdepth: 2
    :hidden:

    get_started
    installation
    user_guide
    examples
    api_reference
    faq
    troubleshooting
    get_involved
    about

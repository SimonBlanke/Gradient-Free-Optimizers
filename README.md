<h1 align="center"> 
  Gradient-Free-Optimizers
</h1>

<h2 align="center">
  Simple and reliable optimization with local, global, population-based and sequential techniques in numerical search spaces.
</h2>

<br>

<table>
  <tbody>
    <tr align="left" valign="center">
      <td>
        <strong>Master status:</strong>
      </td>
      <td>
        <a href="https://travis-ci.com/SimonBlanke/Gradient-Free-Optimizers">
          <img src="https://img.shields.io/travis/com/SimonBlanke/Gradient-Free-Optimizers/master?style=flat-square&logo=travis" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://coveralls.io/github/SimonBlanke/Gradient-Free-Optimizers">
          <img src="https://img.shields.io/coveralls/github/SimonBlanke/Gradient-Free-Optimizers?style=flat-square&logo=codecov" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
    <tr/>
    <tr align="left" valign="center">
      <td>
         <strong>Code quality:</strong>
      </td>
      <td>
        <a href="https://codeclimate.com/github/SimonBlanke/Gradient-Free-Optimizers">
        <img src="https://img.shields.io/codeclimate/maintainability/SimonBlanke/Gradient-Free-Optimizers?style=flat-square&logo=code-climate" alt="img not loaded: try F5 :)">
        </a>
        <a href="https://scrutinizer-ci.com/g/SimonBlanke/Gradient-Free-Optimizers/">
        <img src="https://img.shields.io/scrutinizer/quality/g/SimonBlanke/Gradient-Free-Optimizers?style=flat-square&logo=scrutinizer-ci" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
    <tr/>    <tr align="left" valign="center">
      <td>
        <strong>Latest versions:</strong>
      </td>
      <td>
        <a href="https://pypi.org/project/gradient_free_optimizers/">
          <img src="https://img.shields.io/pypi/v/Gradient-Free-Optimizers?style=flat-square&logo=PyPi&logoColor=white&color=blue" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
  </tbody>
</table>

<br>

## Introduction

Gradient-Free-Optimizers provides a collection of easy to use optimization techniques, 
whose objective function only requires an arbitrary score that gets maximized. 
This makes gradient-free methods capable of solving various optimization problems, including: 
- Optimizing arbitrary mathematical functions.
- Fitting multiple gauss-distributions to data.
- Hyperparameter-optimization of machine-learning methods.


<br>

---

<div align="center"><a name="menu"></a>
  <h3>
    <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers#main-features">Main features</a> •
    <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers#installation">Installation</a> •
    <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers#examples">Examples</a> •
    <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers#basic-api-information">API-info</a> •
    <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers#citation">Citation</a> •
    <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers#license">License</a>
  </h3>
</div>

---

<br>



## Main features

- Easy to use:
  - <a href="https://github.com/SimonBlanke/Gradient-Free-Optimizers#examples">Simple API-design</a>
  - Receive prepared information about ongoing and finished optimization runs

- High performance:
  - Modern optimization techniques
  - Lightweight backend
  - Save time with "short term memory"

- High reliability:
  - Extensive testing
  - Performance test for each optimizer

<br>

### Optimization strategies:
- Local search
  - Hill Climbing
  - Stochastic Hill Climbing
  - Tabu Search
  - Simulated Annealing
- Global search
  - Random Search
  - Random Restart Hill Climbing
  - Random Annealing
- Population methods
  - Parallel Tempering
  - Particle Swarm Optimization
  - Evolution Strategy
- Sequential methods
  - Bayesian Optimization
  - Tree of Parzen Estimators
  - Decision Tree Optimizer


<br>


## Installation

[![PyPI version](https://badge.fury.io/py/gradient-free-optimizers.svg)](https://badge.fury.io/py/gradient-free-optimizers)

The most recent version of Gradient-Free-Optimizers is available on PyPi:

```console
pip install gradient-free-optimizers
```

<br>


## Examples

<details open>
<summary><b>Convex function</b></summary>

```python
import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def parabola_function(para):
    loss = para["x"] * para["x"]
    return -loss


search_space = {"x": np.arange(-10, 10, 0.1)}

opt = RandomSearchOptimizer(search_space)
opt.search(parabola_function, n_iter=100000)
```

</details>


<details>
<summary><b>Non-convex function</b></summary>

```python
import numpy as np
from gradient_free_optimizers import RandomSearchOptimizer


def ackley_function(pos_new):
    x = pos_new["x1"]
    y = pos_new["x2"]

    a1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
    a2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    score = a1 + a2 + 20
    return -score


search_space = {
    "x1": np.arange(-100, 101, 0.1),
    "x2": np.arange(-100, 101, 0.1),
}

opt = RandomSearchOptimizer(search_space)
opt.search(ackley_function, n_iter=30000)
```

</details>


<details>
<summary><b>Machine learning example</b></summary>

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine

from gradient_free_optimizers import HillClimbingOptimizer


data = load_wine()
X, y = data.data, data.target


def model(para):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_space = {
    "n_estimators": np.arange(20, 120, 1),
    "max_depth": np.arange(2, 12, 1),
    "min_samples_split": np.arange(2, 12, 1),
    "min_samples_leaf": np.arange(1, 12, 1),
}

opt = HillClimbingOptimizer(search_space)
opt.search(model, n_iter=50)
```

</details>


<br>

## Basic API-information

### Optimization classes

<p align="center">
  <br>
  <img src="./docs/images/optimizer_table-1.png" width="950">
  <br>
</p>
The table shows the expected results for each optimization strategy for a given type of problem. Those recomendations are just estimated based on personal experience and can heavily change dependend on optimization parameters, exact type of problem and number of iterations.



<details>
<summary><b>HillClimbingOptimizer(...)</b></summary>

Parameter:
  - search_space
    - type: dict

      Creates the n-dimensional search space, where "n" is the number of keys in the dictionary. The values of the dictionary must be numpy arrays. Those arrays determine which numerical values can be chosen in the objective function during the optimization run.

      Examples:
      ```python
      # 1-dimensional search space from 0 to 9
      search_space = {
          "x": np.arange(0, 10, 1),
      }
      ```

      ```python
      # 3-dimensional search space with various ranges
      search_space = {
          "x": np.arange(0, 50, 0.3),
          "y": np.arange(-10, 10, 1),
          "z": np.arange(-100, 100, 0.1),
      }
      ```


  - epsilon
    - type: float (optional, default: 0.05)

      Determines how far the hill climbing based algorithm will "jump" from one point to the next.


  - distribution
    - type: string (optional, default: "normal")

      possible values: 
        - "normal"
        - "laplace"
        - "logistic"
        - "gumbel"

      The (numpy) distribution that decides how to draw samples from the search space in hill climbing based algorithms during the optimization run.


  - n_neighbours
    - type: int (optional, default: 3)

      Hill climbing based algorithms will draw n samples with its distribution and jump to the best of those points. Here "n" is the number of neighbours or "n_neighbours".


  - rand_rest_p
    - type: float (optional, default: 0.01)

      Hill climbing based algorithms tend to get stuck in local optima. To avoid this but still preserve the functionality of the algorithm there is a small percentage in each iteration to jump to a random point in the search space. 



</details>


<details>
<summary><b>StochasticHillClimbingOptimizer(...)</b></summary>

Parameter:
  - search_space
    - type: dict

      Creates the n-dimensional search space, where "n" is the number of keys in the dictionary. The values of the dictionary must be numpy arrays. Those arrays determine which numerical values can be chosen in the objective function during the optimization run.

      Examples:
      ```python
      # 1-dimensional search space from 0 to 9
      search_space = {
          "x": np.arange(0, 10, 1),
      }
      ```

      ```python
      # 3-dimensional search space with various ranges
      search_space = {
          "x": np.arange(0, 50, 0.3),
          "y": np.arange(-10, 10, 1),
          "z": np.arange(-100, 100, 0.1),
      }
      ```


  - epsilon
    - type: float (optional, default: 0.05)

      Determines how far the hill climbing based algorithm will "jump" from one point to the next.


  - distribution
    - type: string (optional, default: "normal")

      possible values: 
        - "normal"
        - "laplace"
        - "logistic"
        - "gumbel"

      The (numpy) distribution that decides how to draw samples from the search space in hill climbing based algorithms during the optimization run.


  - n_neighbours
    - type: int (optional, default: 3)

      Hill climbing based algorithms will draw n samples with its distribution and jump to the best of those points. Here "n" is the number of neighbours or "n_neighbours".


  - rand_rest_p
    - type: float (optional, default: 0.01)

      Hill climbing based algorithms tend to get stuck in local optima. To avoid this but still preserve the functionality of the algorithm there is a small percentage in each iteration to jump to a random point in the search space. 


  - p_accept

    type: float (optional, default: 0.1)


  - norm_factor

    type: string, float (optional, default: "adaptive")

</details>


<details>
<summary><b>TabuOptimizer(...)</b></summary>

Parameter:
  - search_space
    - type: dict

      Creates the n-dimensional search space, where "n" is the number of keys in the dictionary. The values of the dictionary must be numpy arrays. Those arrays determine which numerical values can be chosen in the objective function during the optimization run.

      Examples:
      ```python
      # 1-dimensional search space from 0 to 9
      search_space = {
          "x": np.arange(0, 10, 1),
      }
      ```

      ```python
      # 3-dimensional search space with various ranges
      search_space = {
          "x": np.arange(0, 50, 0.3),
          "y": np.arange(-10, 10, 1),
          "z": np.arange(-100, 100, 0.1),
      }
      ```


  - epsilon
    - type: float (optional, default: 0.05)

      Determines how far the hill climbing based algorithm will "jump" from one point to the next.


  - distribution
    - type: string (optional, default: "normal")

      possible values: 
        - "normal"
        - "laplace"
        - "logistic"
        - "gumbel"

      The (numpy) distribution that decides how to draw samples from the search space in hill climbing based algorithms during the optimization run.


  - n_neighbours
    - type: int (optional, default: 3)

      Hill climbing based algorithms will draw n samples with its distribution and jump to the best of those points. Here "n" is the number of neighbours or "n_neighbours".


  - rand_rest_p
    - type: float (optional, default: 0.01)

      Hill climbing based algorithms tend to get stuck in local optima. To avoid this but still preserve the functionality of the algorithm there is a small percentage in each iteration to jump to a random point in the search space. 


  - tabu_factor

    type: float (optional, default: 3)


</details>


<details>
<summary><b>SimulatedAnnealingOptimizer(...)</b></summary>

Parameter:
  - TODO

</details>


<details>
<summary><b>RandomSearchOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>RandomRestartHillClimbingOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>RandomAnnealingOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>ParallelTemperingOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>ParticleSwarmOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>EvolutionStrategyOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>BayesianOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>TreeStructuredParzenEstimators(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>DecisionTreeOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


<details>
<summary><b>EnsembleOptimizer(...)</b></summary>

Parameter:
  - TODO


</details>


### Optimizer class attributes

  - results
    - (dataframe)
  - best_score
    - (float)
  - best_value
    - (tuple)
  - best_para
    - (dict)
  - eval_times
    - (float)
  - iter_times
    - (float)

</details>


### Optimizer class methods

<details>
<summary><b>search(...)</b></summary>

Search method arguments:
  - objective_function 
    - (callable)
  - n_iter 
    - (int)
  - initialize 
    - (dict, optional)
  - warm_start 
    - (dict, optional)
  - max_time 
    - (float, optional)
  - max_score 
    - (float, optional)
  - memory 
    - (bool, optional)
  - memory_warm_start 
    - (tuple, optional)
  - verbosity 
    - (dict, optional)
  - random_state 
    - (int, optional)

</details>



<br>

### GFOs-design

This package was created as the optimization backend of the Hyperactive package.
The separation of Gradient-Free-Optimizers from Hyperactive enables multiple advantages:
  - Other developers can easily use GFOs as an optimizaton backend if desired
  - Separate and more thorough testing
  - Better isolation from the complex information flow in Hyperactive. GFOs only uses positions and scores in a N-dimensional search-space. It returns only the new position after each iteration.
  - a smaller and cleaner code base, if you want to explore my implementation of these optimization techniques.



<br>

## Citation

    @Misc{gfo2020,
      author =   {{Simon Blanke}},
      title =    {{Gradient-Free-Optimizers}: Simple and reliable optimization with local, global, population-based and sequential techniques in numerical search spaces.},
      howpublished = {\url{https://github.com/SimonBlanke}},
      year = {since 2020}
    }


<br>

## License

Gradient-Free-Optimizers is licensed under the following License:

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Gradient-Free-Optimizers?style=for-the-badge)](https://github.com/SimonBlanke/Gradient-Free-Optimizers/blob/master/LICENSE)



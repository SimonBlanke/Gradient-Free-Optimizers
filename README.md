<h1> Gradient-Free-Optimizers </h1>

<h2 align="center">A collection of gradient free optimizers.</h2>

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
          <img src="https://img.shields.io/pypi/v/Gradient-Free-Optimizers?style=flat-square&logo=PyPi&logoColor=white" alt="img not loaded: try F5 :)">
        </a>
      </td>
    </tr>
  </tbody>
</table>

<br>

## Introduction

Gradient-Free-Optimizers provides a collection of optimization techniques, that do not require the gradient of a given point in the search space to calculate the next one. This makes gradient-free optimization methods capable of performing hyperparameter-optimization of machine learning methods. The optimizers in this package only requires the score of the point to decide which point to evaluate next.

<br>

## GFOs-design

This package was created as the optimization backend of the Hyperactive package. Therefore the API of Gradient-Free-Optimizers is not designed for easy usage. Hyperactive provides a much simpler user experience.
However the separation of Gradient-Free-Optimizers from Hyperactive enables multiple advantages:
  - Other developers can easily use GFOs as an optimizaton backend if desired
  - Separate and more thorough testing
  - Better isolation from the complex information flow in Hyperactive. GFOs only uses positions and scores in a N-dimensional search-space. It returns only the new position after each iteration.

<br>

## Usage

```python
import numpy as np
from gradient_free_optimizers import HillClimbingOptimizer

n_iter = 10

# objective function must be provided by user
def get_score(pos_new):
    x1 = pos_new[0]

    return -x1 * x1

# GFOs must know the dimension of the search space and the initial positions 
space_dim = np.array([100])
init_positions = [np.array([10])]

opt = HillClimbingOptimizer(init_positions, space_dim, opt_para={})

# Initialize the starting positions in this loop
for nth_init in range(len(init_positions)):
    pos_new = opt.init_pos(nth_init)
    score_new = get_score(pos_new) # score must be provided by objective-function
    opt.evaluate(score_new)

# Optimization iteration
for nth_iter in range(len(init_positions), n_iter):
    pos_new = opt.iterate(nth_iter)
    score_new = get_score(pos_new) # score must be provided by objective-function
    opt.evaluate(score_new)
```
